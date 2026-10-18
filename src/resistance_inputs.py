"""Resistance preprocessing with one deterministic physical proxy; public input stays four sensors."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import FEATURES
from .dnn_gami import DNNGAMI
from .dnn_gami import make_bags,distance_hint,fit_count_policy,apply_policy
from .dnn_anomaly import seed_training
from .selective import finite_features
from .utils import save_json,sha256

PROXIES={'resistance':{'column':'resistance_proxy_mohm','unit':'mOhm','formula':'voltage_v / current_ka'}}


def augment_inputs(x,feature):
    """Calculate in original units, before any Train-only standardization."""
    if feature not in PROXIES: raise ValueError('Unknown derived feature')
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or x.shape[1]!=6: raise ValueError('Six raw columns including fixed material context required')
    i,v,t=x[:,3],x[:,4],x[:,5]
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        derived=v/i
    return np.column_stack([x,derived])


class ResistanceInputs(DNNGAMI):
    mode='resistance_preprocessing_base'

    def __init__(self,feature='resistance',**kwargs):
        if feature not in PROXIES: raise ValueError('Unknown derived feature')
        super().__init__(**kwargs)
        self.feature=feature
        self.model_features=[*FEATURES,PROXIES[feature]['column']]

    def normalize(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1] not in [6,7]: raise ValueError('Raw or augmented input matrix required')
        if x.shape[1]==6: x=augment_inputs(x,self.feature)
        return super().normalize(x)

    def score(self,x):
        x=np.asarray(x,dtype=float); normalized=self.normalize(x)
        valid=np.isfinite(normalized).all(1); result=np.full(len(x),np.nan)
        self.network.eval()
        with torch.no_grad():
            for start in range(0,len(x),4096):
                mask=valid[start:start+4096]
                if mask.any():
                    logits=self.network(torch.from_numpy(normalized[start:start+4096][mask])).squeeze(-1)
                    result[np.flatnonzero(mask)+start]=torch.sigmoid(logits).cpu().numpy().astype(float)
        return result

    def fit_counts(self,frame,reference,validation=None,provenance='aggregate_weak_supervision'):
        if provenance not in {'aggregate_weak_supervision','synthetic_verification_only'}: raise ValueError('Explicit provenance required')
        raw,valid=finite_features(frame); x=augment_inputs(raw,self.feature)
        if not len(x) or not valid.all() or not np.isfinite(x).all() or 'date' not in frame: raise ValueError('Valid finite dated rows required')
        self.bags,self.bag_audit=make_bags(frame,reference)
        if not self.bags: raise ValueError('Observed aggregate constraints required')
        seed_training(self.seed)
        self.constant=np.ptp(x,axis=0)==0; self.active=~self.constant
        if int(self.active.sum())!=5: raise ValueError('Four varying sensors plus one varying proxy required')
        self.fixed_values=np.median(x,axis=0); self.mean=x.mean(0); std=x.std(0)
        self.scale=np.where(std>1e-12,std,1.)
        # Keep the original hint. Adding a proxy must not also relabel distances.
        hint,self.hint_statistics=distance_hint(raw)
        tx=torch.from_numpy(self.normalize(x)); th=torch.tensor(hint,dtype=torch.float32)
        if not torch.isfinite(tx).all(): raise ValueError('Normalization overflow')
        rng=np.random.default_rng(self.seed); idx=np.sort(rng.choice(len(x),min(self.anchor_rows,len(x)),replace=False))
        self.anchor_count=len(idx); self.anchor_train_indices=idx.tolist(); self.network=self.build_network()
        self.network.anchors.copy_(tx[idx])
        prior=sum((r['lower']+(r['upper'] if r['upper'] is not None else r['lower']))/2 for r in self.bag_audit)/sum(r['rows'] for r in self.bag_audit)
        prior=float(np.clip(prior,1e-6,1-1e-6)); self.network.bias.data.fill_(float(np.log(prior/(1-prior))))
        self.history=[]; self.stage_selection=[]; self.pruning=[]; self.validation_selection=None
        self._count_phase('main',self.main_epochs,tx,th,frame,reference,None)
        with torch.no_grad():
            rms=self.network.components(tx)['main'].square().mean(0).sqrt().numpy()
            keep=rms>=max(float(rms.max())*self.prune_fraction,1e-8); keep[int(np.argmax(rms))]=True
            self.network.main_mask.copy_(torch.tensor(keep,dtype=torch.float32))
            eligible=[bool(keep[i] or keep[j]) for i,j in self.network.pairs]
            self.network.pair_mask.copy_(torch.tensor(eligible,dtype=torch.float32))
        self.pruning.append({'phase':'main','train_effect_rms':rms.tolist(),'retained':keep.tolist()})
        self._count_phase('pair',self.pair_epochs,tx,th,frame,reference,None)
        with torch.no_grad():
            rms=self.network.components(tx)['pair'].square().mean(0).sqrt().numpy()
            chosen=sorted([k for k,e in enumerate(eligible) if e and rms[k]>=max(float(rms.max())*self.prune_fraction,1e-8)],key=lambda k:(-float(rms[k]),k))[:self.max_interactions]
            mask=torch.zeros(len(self.network.pairs)); mask[chosen]=1
            self.network.pair_mask.copy_(mask); self.network.dnn_enabled.fill_(False)
        self.pruning.append({'phase':'pair','train_effect_rms':rms.tolist(),'retained_indices':chosen,'selection_source':'training_inputs_only'})
        self._count_phase('joint',self.max_epochs,tx,th,frame,reference,validation)
        self.policy=fit_count_policy(self.score(x),frame.date,reference,self.abstention_budget)
        self.fit_rows=len(x); self.fit_labeled_rows=0; self.all_rows_in_loss=True
        self.fit_dates=sorted(frame.date.unique()); self.provenance=provenance
        self.best_epoch=self.stage_selection[-1]['best_epoch']; self.best_validation_loss=None
        self.pseudo_support={'hard_row_labels_used':0,'soft_hint_rows':len(x),'excluded_boundary':0}
        return self

    def predict(self,frame):
        context=frame.copy()
        for k,col in enumerate(FEATURES[:2]):
            if col not in context: context[col]=self.fixed_values[k]
        raw,valid=finite_features(context); x=augment_inputs(raw,self.feature)
        known=np.isclose(raw[:,:2],self.fixed_values[:2],rtol=0,atol=1e-12).all(1)
        usable=valid&known&np.isfinite(x).all(1); score=np.full(len(x),np.nan)
        if usable.any(): score[usable]=self.score(x[usable])
        code=apply_policy(score,self.policy); labels={0:'성공',1:'실패',-1:'모르겠음'}
        reason=np.full(len(x),'학습 집계로 선택한 경계 부근: 판정 보류',dtype=object)
        reason[code==0]='낮은 위험 점수 영역: 성공 추정'; reason[code==1]='높은 위험 점수 영역: 실패 추정'
        reason[valid&~known]='학습하지 않은 고정 두께: 판정 보류'
        reason[~valid|~np.isfinite(x).all(1)]='입력 누락·비유한 값·비양수 값 또는 파생변수 계산 불가: 판정 보류'
        out=pd.DataFrame({'decision_code':code,'decision':[labels[int(c)] for c in code],
            'classification':[labels[int(c)] for c in code],'decision_reason':reason,'anomaly_score':score,
            PROXIES[self.feature]['column']:np.where(valid,x[:,-1],np.nan),
            'prediction_basis':'five_input_GAMI_aggregate_weak_supervision_not_inspection'})
        if 'record_id' in frame: out.insert(0,'record_id',frame.record_id.to_numpy())
        return out

    def save(self,folder):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        weights=folder/'resistance_inputs_weights.pt'; torch.save(self.network.state_dict(),weights)
        keys=['seed','main_hidden','pair_hidden','hidden','main_epochs','pair_epochs','max_epochs','patience','learning_rate','weight_decay',
            'clarity_weight','sparsity_weight','residual_weight','max_interactions','prune_fraction','anchor_rows',
            'hint_weight','consistency_weight','entropy_weight','abstention_budget']
        save_json(folder/'metadata.json',{'inference_mode':self.mode,'feature':self.feature,'proxy':PROXIES[self.feature],
            'raw_feature_columns':FEATURES,'model_feature_columns':self.model_features,'learned_input_columns':self.model_features[2:],
            'public_input_columns':FEATURES[2:],'public_output':['성공','실패','모르겠음'],
            'config':{k:getattr(self,k) for k in keys},'use_dnn':False,
            'mean':self.mean.tolist(),'scale':self.scale.tolist(),'constant':self.constant.tolist(),'fixed_values':self.fixed_values.tolist(),
            'anchor_count':self.anchor_count,'anchor_train_indices':self.anchor_train_indices,'policy':self.policy,
            'fit_rows':self.fit_rows,'fit_dates':self.fit_dates,'all_rows_in_loss':True,'hard_row_labels_used':0,
            'bag_audit':self.bag_audit,'hint_statistics':self.hint_statistics,'hint_uses_derived_feature':False,
            'stage_selection':self.stage_selection,'validation_selection':self.validation_selection,'pruning':self.pruning,
            'training_label_provenance':self.provenance,'labels_are_inspection_truth':False,'score_is_true_defect_probability':False,
            'checkpoint_sha256':sha256(weights)})

