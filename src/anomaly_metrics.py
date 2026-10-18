"""Proxy diagnostics only: the labels are statistical rules, not inspections."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss

from .utils import categorical_labels


def proxy_metrics(y,p,decisions):
    y=categorical_labels(y,(-1,0,1)); d=categorical_labels(decisions,(-1,0,1))
    p=np.asarray(p,dtype=float)
    if p.shape!=y.shape or d.shape!=y.shape: raise ValueError('Proxy evaluation shape mismatch')
    known=y!=-1; finite=np.isfinite(p)&(p>=0)&(p<=1); scored=known&finite
    both=set(y[scored])=={0,1}
    positives=int((y==1).sum())
    caught=int(((y==1)&(d==1)).sum())
    missed=int(((y==1)&(d==0)).sum())
    deferred=int(((y==1)&(d==-1)).sum())
    decided=known&(d!=-1)
    def ratio(a,b): return float(a/b) if b else None
    precision=ratio(caught,int(((d==1)&known).sum()))
    recall=ratio(caught,positives)
    f2=5*precision*recall/(4*precision+recall) if precision is not None and recall is not None and 4*precision+recall else 0. if positives else None
    loss=float(log_loss(y[scored],p[scored],labels=[0,1])) if scored.any() else None
    balanced=None
    if both:
        clipped=np.clip(p[scored],1e-7,1-1e-7); sy=y[scored]
        balanced=float(.5*(-np.log1p(-clipped[sy==0]).mean()-np.log(clipped[sy==1]).mean()))
    return {'rows':len(y),'pseudo_normal_rows':int((y==0).sum()),'pseudo_anomaly_rows':positives,
            'excluded_pseudo_boundary_rows':int((~known).sum()),'pseudo_scored_rows':int(scored.sum()),
            'pseudo_pr_auc':float(average_precision_score(y[scored],p[scored])) if both else None,
            'pseudo_roc_auc':float(roc_auc_score(y[scored],p[scored])) if both else None,
            'pseudo_log_loss':loss,'pseudo_balanced_bce':balanced,
            'pseudo_binary_accuracy_at_0_5':float(((p[scored]>=.5)==y[scored]).mean()) if scored.any() else None,
            'pseudo_decision_precision':precision,'pseudo_recall_including_abstentions':recall,'pseudo_decision_f2':f2,
            'pseudo_anomalies_detected':caught,'pseudo_anomalies_predicted_success':missed,
            'pseudo_anomalies_abstained':deferred,
            'pseudo_selective_error':ratio(int(((d!=y)&decided).sum()),int(decided.sum())),
            'success_outputs':int((d==0).sum()),'defect_candidate_outputs':int((d==1).sum()),
            'unknown_outputs':int((d==-1).sum()),'decision_coverage':float((d!=-1).mean()),
            'abstention_rate':float((d==-1).mean())}
