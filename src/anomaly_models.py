"""Per-weld statistical anomaly inference; no inspected quality labels implied."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES,OUTPUTS
from .selective import finite_features
from .utils import save_json,sha256


class AlwaysSuccess:
    """The user's untrained Baseline: every record is predicted successful."""
    def predict(self,frame):
        out=pd.DataFrame({'decision_code':np.zeros(len(frame),dtype=int),
                          'decision':['성공']*len(frame),
                          'decision_reason':['학습 없는 Baseline']*len(frame),
                          'anomaly_score':np.zeros(len(frame))})
        if 'record_id' in frame: out.insert(0,'record_id',frame.record_id.to_numpy())
        return out


class LogisticAnomaly:
    """Learn a sigmoid of robust radial distance from statistical pseudo labels.

    Four variable sensors form an ellipsoidal boundary in original units.
    Logistic probabilities refer to pseudo classes, not true welding defects.
    """
    mode='statistical_pseudo_label_anomaly'

    def __init__(self,normal_quantile=.95,anomaly_quantile=.995,
                 success_threshold=.2,defect_threshold=.8):
        if not 0<normal_quantile<anomaly_quantile<1:
            raise ValueError('Ordered pseudo-label quantiles required')
        if not 0<success_threshold<defect_threshold<1:
            raise ValueError('Ordered decision thresholds required')
        self.normal_quantile,self.anomaly_quantile=normal_quantile,anomaly_quantile
        self.success_threshold,self.defect_threshold=success_threshold,defect_threshold

    def fit(self,frame):
        x,valid=finite_features(frame)
        if not len(x) or not valid.all(): raise ValueError('Finite positive training inputs required')
        self.center=np.median(x,axis=0)
        self.constant=np.ptp(x,axis=0)==0
        self.active=~self.constant
        if not self.active.any(): raise ValueError('No variable sensors to learn')
        q1,q3=np.quantile(x,[.25,.75],axis=0)
        scale=q3-q1
        fallback=np.std(x,axis=0)
        self.scale=np.where(scale>1e-12,scale,np.where(fallback>1e-12,fallback,1.))
        d=self.distance_squared(x)
        self.normal_cut,self.anomaly_cut=np.quantile(d,[self.normal_quantile,self.anomaly_quantile])
        if self.normal_cut>=self.anomaly_cut:
            raise ValueError('Training distance distribution cannot separate pseudo classes')
        pseudo=np.full(len(x),-1,dtype=int)
        pseudo[d<=self.normal_cut]=0
        pseudo[d>=self.anomaly_cut]=1
        used=pseudo!=-1
        self.estimator=make_pipeline(StandardScaler(),LogisticRegression(
            C=1.,class_weight='balanced',max_iter=1000,random_state=42))
        self.estimator.fit(d[used,None],pseudo[used])
        if self.estimator[-1].coef_[0,0]<=0:
            raise ValueError('Anomaly score must increase with distance')
        self.fit_rows=len(x)
        self.pseudo_support={'normal':int((pseudo==0).sum()),'anomaly':int((pseudo==1).sum()),
                             'excluded_boundary':int((pseudo==-1).sum())}
        self.fit_pseudo_labels=pseudo
        return self

    def distance_squared(self,x):
        x=np.asarray(x,dtype=float)
        with np.errstate(over='ignore',invalid='ignore'):
            z=(x[:,self.active]-self.center[self.active])/self.scale[self.active]
            return np.mean(z*z,axis=1)

    def score(self,x):
        d=self.distance_squared(x)
        # Saturate overflow from otherwise finite, extremely distant inputs.
        result=np.ones(len(d),dtype=float)
        finite=np.isfinite(d)
        if finite.any(): result[finite]=self.estimator.predict_proba(d[finite,None])[:,1]
        return result

    def predict(self,frame):
        x,valid=finite_features(frame)
        known_constants=np.isclose(x[:,self.constant],self.center[self.constant],rtol=0,atol=1e-12).all(1)
        usable=valid&known_constants
        p=np.full(len(x),np.nan)
        if usable.any(): p[usable]=self.score(x[usable])
        decision=np.full(len(x),-1,dtype=int)
        decision[usable&(p<=self.success_threshold)]=0
        decision[usable&(p>=self.defect_threshold)]=1
        reason=np.full(len(x),'정상·이상 경계 부근: 판정 보류',dtype=object)
        reason[decision==0]='정상 분포 영역: 성공 추정'
        reason[decision==1]='분포에서 크게 벗어남: 불량 추정'
        reason[valid&~known_constants]='학습하지 않은 고정 입력값(두께 등): 판정 보류'
        reason[~valid]='입력 누락·비유한 값·비양수 값: 판정 보류'
        out=pd.DataFrame({'decision_code':decision,'decision':[OUTPUTS[int(i)] for i in decision],
                          'decision_reason':reason,'anomaly_score':p,
                          'distance_squared':self.distance_squared(x),
                          'prediction_basis':['statistical_anomaly_not_inspection']*len(x)})
        if 'record_id' in frame: out.insert(0,'record_id',frame.record_id.to_numpy())
        return out

    def save(self,folder):
        folder=Path(folder)
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/'logistic_anomaly.joblib'
        joblib.dump(self,path)
        save_json(folder/'metadata.json',{'inference_mode':self.mode,'model':'Logistic',
            'feature_columns':FEATURES,'active_features':[f for f,a in zip(FEATURES,self.active) if a],
            'fit_rows':self.fit_rows,'center':self.center.tolist(),'iqr_or_std_scale':self.scale.tolist(),
            'pseudo_label_quantiles':[self.normal_quantile,self.anomaly_quantile],
            'distance_squared_pseudo_cutoffs':[float(self.normal_cut),float(self.anomaly_cut)],
            'decision_thresholds':[self.success_threshold,self.defect_threshold],
            'pseudo_support':self.pseudo_support,'labels_are_inspection_truth':False,
            'score_is_true_defect_probability':False,'checkpoint_sha256':sha256(path)})

    @classmethod
    def load(cls,folder):
        folder=Path(folder)
        info=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        if info.get('inference_mode')!=cls.mode or info['feature_columns']!=FEATURES:
            raise ValueError('Anomaly checkpoint mode or inputs mismatch')
        path=folder/'logistic_anomaly.joblib'
        if sha256(path)!=info['checkpoint_sha256']: raise ValueError('Checkpoint checksum mismatch')
        return joblib.load(path)
