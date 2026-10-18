"""DNN_GAMI: all-row interval LLP with a weak distance hint and fixed policy.

Individual sensor scores remain estimates, not inspected quality probabilities.
Dates are used only to form training bags; they are never network inputs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import FEATURES,OUTPUTS
from .gami_components import GAMIBase
from .dnn_anomaly import seed_training
from .selective import finite_features
from .utils import save_json,sha256


def distance_hint(x):
    center=np.median(x,axis=0); active=np.ptp(x,axis=0)>0
    q1,q3=np.quantile(x,[.25,.75],axis=0); std=x.std(0)
    scale=np.where(q3-q1>1e-12,q3-q1,np.where(std>1e-12,std,1.))
    d=np.mean(((x[:,active]-center[active])/scale[active])**2,axis=1)
    lo,hi=np.quantile(d,[.95,.995])
    target=np.clip((d-lo)/max(hi-lo,1e-12),0,1)
    return target,{'center':center.tolist(),'scale':scale.tolist(),'normal_cut':float(lo),
                   'anomaly_cut':float(hi),'active':active.tolist()}


def interval_loss(probabilities,bags):
    terms=[]
    for indices,lo,hi in bags:
        count=probabilities[indices].sum()
        gap=nn.functional.relu(count.new_tensor(lo)-count)
        if hi is not None: gap=gap+nn.functional.relu(count-count.new_tensor(hi))
        terms.append((gap/max(hi if hi is not None else lo,1.))**2)
    if not terms: raise ValueError('At least one observed training count is required')
    return torch.stack(terms).mean()


def make_bags(frame,reference):
    ref=reference.set_index('date'); bags=[]; audit=[]
    for date in sorted(frame.date.unique()):
        if date not in ref.index: continue
        row=ref.loc[date]; lo,hi=row.unique_defect_count_lower,row.unique_defect_count_upper
        indices=np.flatnonzero(frame.date.eq(date).to_numpy())
        if pd.isna(lo): continue
        if lo<0 or lo>len(indices) or (pd.notna(hi) and (hi<lo or hi>len(indices))):
            raise ValueError('Count interval inconsistent with the training bag')
        upper=None if pd.isna(hi) else float(hi)
        bags.append((torch.tensor(indices,dtype=torch.long),float(lo),upper))
        audit.append({'date':date,'rows':len(indices),'lower':float(lo),'upper':upper})
    return bags,audit


def count_errors(scores,dates,reference,cut):
    rows=[]
    for row in reference.to_dict('records'):
        mask=np.asarray(dates)==row['date']
        lo,hi=row['unique_defect_count_lower'],row['unique_defect_count_upper']
        if not mask.any() or pd.isna(lo) or pd.isna(hi): continue
        hard=int((scores[mask]>=cut).sum()); soft=float(scores[mask].sum())
        rows.append({'date':row['date'],'hard':hard,'soft':soft,
                     'interval_gap':float(max(lo-hard,0,hard-hi)),
                     'sum_gap':float(abs(hard-row['reported_type_sum'])),
                     'soft_gap':float(max(lo-soft,0,soft-hi))})
    if not rows: raise ValueError('Complete count intervals required to select a decision policy')
    return {'interval_mae':float(np.mean([r['interval_gap'] for r in rows])),
            'type_sum_mae':float(np.mean([r['sum_gap'] for r in rows])),
            'soft_interval_mae':float(np.mean([r['soft_gap'] for r in rows]))}


def fit_count_policy(scores,dates,reference,abstention_budget=.01):
    """One global rule fitted on training dates. Never per-day top-k selection."""
    scores=np.asarray(scores,dtype=float); dates=np.asarray(dates)
    if scores.ndim!=1 or len(scores)!=len(dates) or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite aligned policy scores required')
    if (scores<0).any() or (scores>1).any() or not 0<=abstention_budget<1:
        raise ValueError('Invalid policy scores or budget')
    values=np.unique(scores)
    cuts=np.r_[np.nextafter(values[0],-np.inf),(values[:-1]+values[1:])/2,np.nextafter(values[-1],np.inf)]
    ref=reference.loc[reference.date.isin(dates)&reference.unique_defect_count_lower.notna()&reference.unique_defect_count_upper.notna()]
    if not len(ref): raise ValueError('Complete training intervals required')
    gaps=[]; sums=[]
    for row in ref.to_dict('records'):
        observed=np.sort(scores[dates==row['date']]); counts=len(observed)-np.searchsorted(observed,cuts,side='left')
        gaps.append(np.maximum.reduce([row['unique_defect_count_lower']-counts,np.zeros(len(cuts)),counts-row['unique_defect_count_upper']]))
        sums.append(abs(counts-row['reported_type_sum']))
    gap=np.mean(gaps,axis=0); total_gap=np.mean(sums,axis=0)
    index=min(range(len(cuts)),key=lambda k:(gap[k],total_gap[k],-cuts[k]))
    upper=float(cuts[index]); lower=max(0.,upper*.9)
    # Narrow the band when duplicate scores would exceed the training budget.
    ordered=np.sort(scores); at=np.searchsorted(ordered,upper,side='left'); allowed=int(len(scores)*abstention_budget)
    if int(((scores>=lower)&(scores<upper)).sum())>allowed:
        lower=float(np.nextafter(ordered[max(0,at-allowed-1)],np.inf)) if at else upper
        lower=min(lower,upper)
    unknown=float(((scores>=lower)&(scores<upper)).mean())
    return {'success_below':lower,'defect_at_or_above':upper,'training_abstention_rate':unknown,
            'training_interval_mae':float(gap[index]),'training_type_sum_mae':float(total_gap[index]),
            'abstention_budget':abstention_budget,'fit_dates':sorted(ref.date.unique()),
            'inference_uses_date_or_count':False,'is_quality_probability_calibration':False}


def apply_policy(scores,policy):
    p=np.asarray(scores,dtype=float); result=np.full(len(p),-1,dtype=int)
    valid=np.isfinite(p)&(p>=0)&(p<=1)
    result[valid&(p<policy['success_below'])]=0
    result[valid&(p>=policy['defect_at_or_above'])]=1
    return result


class DNNGAMI(GAMIBase):
    display_name='DNN_GAMI'
    mode='DNN_GAMI_all_row_aggregate_weak_supervision'

    def __init__(self,seed=42,hint_weight=.01,consistency_weight=.1,entropy_weight=.001,
                 abstention_budget=.01,**kwargs):
        super().__init__(seed=seed,use_dnn=True,**kwargs)
        if min(hint_weight,consistency_weight,entropy_weight)<0 or not 0<=abstention_budget<1:
            raise ValueError('Invalid DNN_GAMI weights')
        self.hint_weight,self.consistency_weight,self.entropy_weight=hint_weight,consistency_weight,entropy_weight
        self.abstention_budget=abstention_budget

    def fit_counts(self,frame,reference,validation=None,provenance='aggregate_weak_supervision'):
        if provenance not in {'aggregate_weak_supervision','synthetic_verification_only'}:
            raise ValueError('Explicit aggregate or fixture provenance required')
        x,valid=finite_features(frame)
        if not len(x) or not valid.all() or 'date' not in frame: raise ValueError('Valid dated sensor rows required')
        self.bags,self.bag_audit=make_bags(frame,reference)
        if not self.bags: raise ValueError('No aggregate constraints in fit dates')
        seed_training(self.seed)
        self.constant=np.ptp(x,axis=0)==0; self.active=~self.constant
        if int(self.active.sum())!=4: raise ValueError('Four variable sensors required')
        self.fixed_values=np.median(x,axis=0); self.mean=x.mean(0); std=x.std(0)
        self.scale=np.where(std>1e-12,std,1.)
        hint,self.hint_statistics=distance_hint(x)
        tx=torch.from_numpy(self.normalize(x)); th=torch.tensor(hint,dtype=torch.float32)
        if not torch.isfinite(tx).all(): raise ValueError('Normalization overflow')
        rng=np.random.default_rng(self.seed); idx=np.sort(rng.choice(len(x),min(self.anchor_rows,len(x)),replace=False))
        self.anchor_count=len(idx); self.anchor_train_indices=idx.tolist(); self.network=self.build_network()
        self.network.anchors.copy_(tx[idx])
        prior=sum((r['lower']+(r['upper'] if r['upper'] is not None else r['lower']))/2 for r in self.bag_audit)/sum(r['rows'] for r in self.bag_audit)
        prior=float(np.clip(prior,1e-6,1-1e-6))
        self.network.bias.data.fill_(float(np.log(prior/(1-prior))))
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
            self.network.pair_mask.copy_(mask); self.network.dnn_enabled.fill_(True)
        self.pruning.append({'phase':'pair','train_effect_rms':rms.tolist(),'retained_indices':chosen,'selection_source':'training_inputs_only'})
        self._count_phase('joint',self.max_epochs,tx,th,frame,reference,validation)
        self.policy=fit_count_policy(self.score(x),frame.date,reference,self.abstention_budget)
        self.fit_rows=len(x); self.fit_labeled_rows=0; self.all_rows_in_loss=True
        self.fit_dates=sorted(frame.date.unique()); self.provenance=provenance
        self.best_epoch=self.stage_selection[-1]['best_epoch']
        self.best_validation_loss=None; self.pseudo_support={'hard_row_labels_used':0,'soft_hint_rows':len(x),'excluded_boundary':0}
        return self

    def _count_phase(self,name,limit,tx,hint,frame,reference,validation):
        for p in self.network.parameters(): p.requires_grad_(False)
        modules=[self.network.main] if name=='main' else [self.network.interactions] if name=='pair' else [self.network.main,self.network.interactions,self.network.dnn]
        for module in modules:
            for p in module.parameters(): p.requires_grad_(True)
        self.network.bias.requires_grad_(True)
        if name=='joint': self.network.alpha_logit.requires_grad_(True)
        optimizer=torch.optim.Adam([p for p in self.network.parameters() if p.requires_grad],lr=self.learning_rate,weight_decay=self.weight_decay)
        best_key=None; best_state=None; best_epoch=0
        noise_generator=torch.Generator().manual_seed(self.seed+1000+len(self.stage_selection))
        for epoch in range(1,limit+1):
            self.network.train(); optimizer.zero_grad()
            c=self.network.components(tx); z=c['bias']+c['main'].sum(1)+c['pair'].sum(1)+c['dnn']; p=torch.sigmoid(z)
            count=interval_loss(p,self.bags)
            weak=nn.functional.binary_cross_entropy_with_logits(z,hint)
            # All rows participate; perturbations are small in standardized units.
            noise=torch.randn(tx.shape,generator=noise_generator)*.01
            perturbed=torch.sigmoid(self.network(tx+noise).squeeze(-1))
            consistency=(p-perturbed).square().mean()
            entropy=-(p*torch.log(p.clamp_min(1e-8))+(1-p)*torch.log((1-p).clamp_min(1e-8))).mean()
            reg=self.network.penalty(c,self.clarity_weight,self.sparsity_weight,self.residual_weight)
            loss=count+self.hint_weight*weak+self.consistency_weight*consistency+self.entropy_weight*entropy+reg
            if not torch.isfinite(loss): raise ValueError('Nonfinite aggregate objective')
            loss.backward(); optimizer.step(); self.network.eval()
            item={'phase':name,'epoch':epoch,'global_epoch':len(self.history)+1,'total_loss':float(loss.detach()),
                  'count_loss':float(count.detach()),'weak_hint_loss':float(weak.detach()),
                  'consistency_loss':float(consistency.detach()),'entropy_loss':float(entropy.detach()),'regularization':float(reg.detach())}
            key=(item['total_loss'],)
            if validation is not None:
                if epoch%10 and epoch!=limit:
                    self.history.append(item); continue
                scores=self.score(frame[FEATURES]); policy=fit_count_policy(scores,frame.date,reference,self.abstention_budget)
                vscore=self.score(validation[FEATURES]); err=count_errors(vscore,validation.date,reference,policy['defect_at_or_above'])
                unknown=float((apply_policy(vscore,policy)==-1).mean())
                key=(err['interval_mae'],err['type_sum_mae'],max(unknown-self.abstention_budget,0),err['soft_interval_mae'],item['total_loss'])
                item.update(validation_interval_mae=err['interval_mae'],validation_type_sum_mae=err['type_sum_mae'],
                            validation_soft_interval_mae=err['soft_interval_mae'],validation_abstention_rate=unknown)
            self.history.append(item)
            if best_key is None or key<best_key:
                best_key=key; best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in self.network.state_dict().items()}
        self.network.load_state_dict(best_state); self.network.eval()
        self.stage_selection.append({'phase':name,'best_epoch':best_epoch,'epochs_run':limit,'selection_key':list(best_key)})
        if validation is not None: self.validation_selection={'key':list(best_key),'dates':sorted(validation.date.unique()),'joint_epoch':best_epoch}
        print(f'{self.display_name} seed {self.seed} {name}: epoch {best_epoch}/{limit}, criterion {best_key}',flush=True)

    def predict(self,frame):
        x,valid=finite_features(frame)
        known=np.isclose(x[:,self.constant],self.fixed_values[self.constant],rtol=0,atol=1e-12).all(1)
        scores=np.full(len(x),np.nan); usable=valid&known
        if usable.any(): scores[usable]=self.score(x[usable])
        codes=apply_policy(scores,self.policy)
        reason=np.array(['학습 집계로 선택한 경계 부근: 판정 보류']*len(x),dtype=object)
        reason[codes==0]='DNN_GAMI 낮은 위험 점수 영역: 성공 추정'; reason[codes==1]='DNN_GAMI 높은 위험 점수 영역: 불량 추정'
        reason[valid&~known]='학습하지 않은 고정 두께: 판정 보류'; reason[~valid]='입력 누락·비유한 값·비양수 값: 판정 보류'
        out=pd.DataFrame({'decision_code':codes,'decision':[OUTPUTS[int(c)] for c in codes],
                          'decision_reason':reason,'anomaly_score':scores,'prediction_basis':'aggregate_weak_supervision_not_inspection'})
        if 'record_id' in frame: out.insert(0,'record_id',frame.record_id.to_numpy())
        return out

    def save(self,folder):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/'dnn_gami_weights.pt'; torch.save(self.network.state_dict(),checkpoint)
        keys=['seed','main_hidden','pair_hidden','hidden','main_epochs','pair_epochs','max_epochs','patience','learning_rate','weight_decay',
              'clarity_weight','sparsity_weight','residual_weight','max_interactions','prune_fraction','anchor_rows',
              'hint_weight','consistency_weight','entropy_weight','abstention_budget']
        info={'inference_mode':self.mode,'model':'DNN_GAMI','feature_columns':FEATURES,'config':{k:getattr(self,k) for k in keys},
              'mean':self.mean.tolist(),'scale':self.scale.tolist(),'constant':self.constant.tolist(),'fixed_values':self.fixed_values.tolist(),
              'anchor_count':self.anchor_count,'anchor_train_indices':self.anchor_train_indices,'policy':self.policy,
              'fit_rows':self.fit_rows,'fit_dates':self.fit_dates,'all_rows_in_loss':True,'hard_row_labels_used':0,
              'bag_audit':self.bag_audit,'hint_statistics':self.hint_statistics,'stage_selection':self.stage_selection,
              'validation_selection':self.validation_selection,'pruning':self.pruning,'training_label_provenance':self.provenance,
              'labels_are_inspection_truth':False,'score_is_true_defect_probability':False,'checkpoint_sha256':sha256(checkpoint)}
        save_json(folder/'metadata.json',info)

    @classmethod
    def load(cls,folder,allow_fixture=False):
        folder=Path(folder); info=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        if info['inference_mode']!=cls.mode or info['feature_columns']!=FEATURES: raise ValueError('DNN_GAMI checkpoint mismatch')
        if info['training_label_provenance']=='synthetic_verification_only' and not allow_fixture: raise ValueError('Synthetic fixture rejected')
        path=folder/'dnn_gami_weights.pt'
        if sha256(path)!=info['checkpoint_sha256']: raise ValueError('DNN_GAMI checkpoint checksum mismatch')
        model=cls(**info['config'])
        for key in ['mean','scale','fixed_values']: setattr(model,key,np.asarray(info[key],dtype=float))
        model.constant=np.asarray(info['constant'],dtype=bool); model.active=~model.constant
        for key in ['anchor_count','anchor_train_indices','policy','fit_rows','fit_dates','bag_audit','hint_statistics','stage_selection','validation_selection','pruning']:
            setattr(model,key,info[key])
        model.network=model.build_network(); model.network.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)); model.network.eval()
        model.all_rows_in_loss=True; model.provenance=info['training_label_provenance']
        return model
