"""A model: fully connected DNN learning statistical per-weld pseudo classes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import FEATURES,OUTPUTS
from .selective import finite_features
from .utils import categorical_labels,save_json,sha256


def seed_training(seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def balanced_bce(logits,labels):
    loss=nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    return .5*(loss[labels==0].mean()+loss[labels==1].mean())


def pseudo_labels(reference,x):
    """Apply the frozen stage-2 distance cutoffs, never held-out quantiles."""
    distance=reference.distance_squared(np.asarray(x,dtype=float))
    return np.where(distance<=reference.normal_cut,0,
                    np.where(distance>=reference.anomaly_cut,1,-1)).astype(int)


class DNNAnomaly:
    mode='dnn_statistical_pseudo_label_anomaly'

    def __init__(self,seed=42,hidden=(32,16),max_epochs=400,patience=40,
                 learning_rate=.003,weight_decay=.0001,
                 success_threshold=.2,defect_threshold=.8):
        if not hidden or any(int(n)!=n or n<1 for n in hidden): raise ValueError('Positive hidden widths required')
        if max_epochs<1 or patience<1: raise ValueError('Positive training limits required')
        if not 0<success_threshold<defect_threshold<1: raise ValueError('Ordered thresholds required')
        self.seed,self.hidden=int(seed),tuple(hidden)
        self.max_epochs,self.patience=int(max_epochs),int(patience)
        self.learning_rate,self.weight_decay=float(learning_rate),float(weight_decay)
        self.success_threshold,self.defect_threshold=success_threshold,defect_threshold

    def build_network(self):
        layers=[]
        widths=[int(self.active.sum()),*self.hidden]
        for a,b in zip(widths[:-1],widths[1:]): layers.extend([nn.Linear(a,b),nn.ReLU()])
        layers.append(nn.Linear(widths[-1],1))
        return nn.Sequential(*layers)

    def normalize(self,x):
        with np.errstate(over='ignore',invalid='ignore'):
            return ((np.asarray(x,dtype=float)[:,self.active]-self.mean[self.active])/
                    self.scale[self.active]).astype(np.float32)

    def fit(self,train_frame,train_y,val_frame,val_y,provenance='statistical_pseudo_labels'):
        if provenance not in {'statistical_pseudo_labels','synthetic_verification_only'}:
            raise ValueError('Explicit pseudo-label or test-fixture provenance required')
        x,valid=finite_features(train_frame)
        vx,vvalid=finite_features(val_frame)
        y=categorical_labels(train_y,(-1,0,1)); vy=categorical_labels(val_y,(-1,0,1))
        if len(x)!=len(y) or len(vx)!=len(vy) or not valid.all() or not vvalid.all():
            raise ValueError('Finite positive inputs and aligned pseudo labels required')
        used,vused=y!=-1,vy!=-1
        if set(y[used])!={0,1} or set(vy[vused])!={0,1}:
            raise ValueError('Both pseudo classes required in training and validation')
        seed_training(self.seed)
        self.constant=np.ptp(x,axis=0)==0; self.active=~self.constant
        if not self.active.any(): raise ValueError('No variable sensors to learn')
        self.fixed_values=np.median(x,axis=0)
        self.mean=x.mean(0); std=x.std(0)
        self.scale=np.where(std>1e-12,std,1.)
        tx=torch.from_numpy(self.normalize(x[used])); ty=torch.tensor(y[used],dtype=torch.float32)
        tvx=torch.from_numpy(self.normalize(vx[vused])); tvy=torch.tensor(vy[vused],dtype=torch.float32)
        if not torch.isfinite(tx).all() or not torch.isfinite(tvx).all(): raise ValueError('Input normalization overflow')
        self.network=self.build_network()
        optimizer=torch.optim.Adam(self.network.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)
        self.history=[]; best=float('inf'); stale=0; best_state=None
        for epoch in range(1,self.max_epochs+1):
            self.network.train(); optimizer.zero_grad()
            loss=balanced_bce(self.network(tx).squeeze(-1),ty)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); optimizer.step()
            self.network.eval()
            with torch.no_grad(): val_loss=float(balanced_bce(self.network(tvx).squeeze(-1),tvy))
            if not np.isfinite(val_loss): raise ValueError('Nonfinite validation loss')
            self.history.append({'epoch':epoch,'train_balanced_bce':float(loss.detach()),
                                 'validation_balanced_bce':val_loss})
            if val_loss<best-1e-5:
                best=val_loss; stale=0; self.best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in self.network.state_dict().items()}
            else: stale+=1
            if stale>=self.patience: break
        self.network.load_state_dict(best_state); self.network.eval()
        self.best_validation_loss=best; self.provenance=provenance
        self.fit_rows=len(x); self.fit_labeled_rows=int(used.sum())
        self.pseudo_support={'normal':int((y==0).sum()),'anomaly':int((y==1).sum()),
                             'excluded_boundary':int((y==-1).sum())}
        return self

    def score(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(FEATURES): raise ValueError('Six numerical input columns required')
        normalized=self.normalize(x)
        valid=np.isfinite(normalized).all(1)
        result=np.full(len(x),np.nan)
        self.network.eval()
        with torch.no_grad():
            for start in range(0,len(x),4096):
                mask=valid[start:start+4096]
                if mask.any():
                    logits=self.network(torch.from_numpy(normalized[start:start+4096][mask])).squeeze(-1)
                    result[np.flatnonzero(mask)+start]=torch.sigmoid(logits).cpu().numpy().astype(float)
        return result

    def predict(self,frame):
        x,valid=finite_features(frame)
        known=np.isclose(x[:,self.constant],self.fixed_values[self.constant],rtol=0,atol=1e-12).all(1)
        usable=valid&known; p=np.full(len(x),np.nan)
        if usable.any(): p[usable]=self.score(x[usable])
        decision=np.full(len(x),-1,dtype=int)
        usable=usable&np.isfinite(p)
        decision[usable&(p<=self.success_threshold)]=0
        decision[usable&(p>=self.defect_threshold)]=1
        reason=np.full(len(x),'DNN 정상·이상 경계 부근: 판정 보류',dtype=object)
        reason[decision==0]='DNN 정상 임시 클래스 영역: 성공 추정'
        reason[decision==1]='DNN 이상 임시 클래스 영역: 불량 추정'
        reason[valid&~known]='학습하지 않은 고정 입력값(두께 등): 판정 보류'
        reason[~valid]='입력 누락·비유한 값·비양수 값: 판정 보류'
        reason[valid&known&~np.isfinite(p)]='DNN 점수 계산 범위 초과: 판정 보류'
        out=pd.DataFrame({'decision_code':decision,'decision':[OUTPUTS[int(i)] for i in decision],
                          'decision_reason':reason,'anomaly_score':p,
                          'prediction_basis':['statistical_pseudo_label_dnn_not_inspection']*len(x)})
        if 'record_id' in frame: out.insert(0,'record_id',frame.record_id.to_numpy())
        return out

    def save(self,folder):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/'dnn_weights.pt'
        torch.save(self.network.state_dict(),checkpoint)
        info={'inference_mode':self.mode,'model':'DNN','feature_columns':FEATURES,
              'config':{'seed':self.seed,'hidden':list(self.hidden),'max_epochs':self.max_epochs,'patience':self.patience,
                        'learning_rate':self.learning_rate,'weight_decay':self.weight_decay,
                        'success_threshold':self.success_threshold,'defect_threshold':self.defect_threshold},
              'constant':self.constant.tolist(),'mean':self.mean.tolist(),'scale':self.scale.tolist(),
              'fixed_values':self.fixed_values.tolist(),'fit_rows':self.fit_rows,'fit_labeled_rows':self.fit_labeled_rows,
              'pseudo_support':self.pseudo_support,'best_epoch':self.best_epoch,
              'best_validation_loss':self.best_validation_loss,'training_label_provenance':self.provenance,
              'labels_are_inspection_truth':False,'score_is_true_defect_probability':False,
              'checkpoint_sha256':sha256(checkpoint)}
        save_json(folder/'metadata.json',info)

    @classmethod
    def load(cls,folder,allow_fixture=False):
        folder=Path(folder); info=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        if info['inference_mode']!=cls.mode or info['feature_columns']!=FEATURES: raise ValueError('DNN model mode or inputs mismatch')
        if info['training_label_provenance']=='synthetic_verification_only' and not allow_fixture:
            raise ValueError('Synthetic fixture model cannot be used as a welding model')
        checkpoint=folder/'dnn_weights.pt'
        if sha256(checkpoint)!=info['checkpoint_sha256']: raise ValueError('DNN checkpoint checksum mismatch')
        model=cls(**info['config'])
        for key in ['mean','scale','fixed_values']: setattr(model,key,np.asarray(info[key],dtype=float))
        model.constant=np.asarray(info['constant'],dtype=bool); model.active=~model.constant
        model.network=model.build_network()
        model.network.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)); model.network.eval()
        for key in ['fit_rows','fit_labeled_rows','pseudo_support','best_epoch','best_validation_loss']:
            setattr(model,key,info[key])
        model.provenance=info['training_label_provenance']
        return model
