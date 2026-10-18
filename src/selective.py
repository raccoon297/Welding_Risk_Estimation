"""Class-conditional conformal sets for binary quality with abstention.

Scores come from a trained binary classifier. The policy is fitted solely to
real inspection labels in the reserved calibration split. Empty/both-label
sets, invalid inputs, and inputs outside the observed training range abstain.
"""
import numpy as np
import pandas as pd

from .config import FEATURES,OUTPUTS
from .utils import categorical_labels


def finite_features(frame):
    x = frame.reindex(columns=FEATURES).apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1) & (x>0).all(axis=1)
    return x,valid


def conformal_threshold(scores,alpha):
    if not 0<alpha<1:
        raise ValueError("alpha must be in (0,1)")
    scores = np.sort(np.asarray(scores,dtype=float))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite calibration scores")
    rank = int(np.ceil((len(scores)+1)*(1-alpha)))
    # A class with inadequate calibration support remains possible everywhere.
    return 1. if rank>len(scores) else float(scores[rank-1])


class SelectivePolicy:
    def __init__(self,alpha=.1):
        self.alpha = float(alpha)

    def fit(self,train_x,calibration_p,calibration_y):
        x = np.asarray(train_x,dtype=float)
        p,y = np.asarray(calibration_p,dtype=float),categorical_labels(calibration_y)
        if x.ndim!=2 or x.shape[1]!=len(FEATURES) or not len(x) or not np.isfinite(x).all() or not (x>0).all():
            raise ValueError("Invalid training inputs")
        if p.shape!=y.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
            raise ValueError("Real binary calibration labels and finite scores required")
        self.minimum,self.maximum = x.min(0),x.max(0)
        self.q_success = conformal_threshold(p[y==0],self.alpha)
        self.q_defect = conformal_threshold(1-p[y==1],self.alpha)
        self.support = {"success":int((y==0).sum()),"defect":int((y==1).sum())}
        return self

    def classify(self,x,p,valid=None):
        x,p = np.asarray(x,dtype=float),np.asarray(p,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(FEATURES) or p.shape!=(len(x),):
            raise ValueError("Input shape mismatch")
        if valid is None:
            valid = np.isfinite(x).all(1)&(x>0).all(1)
        valid = np.asarray(valid,dtype=bool) & np.isfinite(p) & (p>=0)&(p<=1)
        inside = ((x>=self.minimum-1e-12)&(x<=self.maximum+1e-12)).all(1)
        allow_success,allow_defect = p<=self.q_success,1-p<=self.q_defect
        decision = np.full(len(x),-1,dtype=int)
        usable = valid & inside
        decision[usable & allow_success & ~allow_defect] = 0
        decision[usable & allow_defect & ~allow_success] = 1
        reason = np.full(len(x),"불확실한 예측 집합",dtype=object)
        reason[decision==0] = "보정된 예측 집합에 성공만 포함"
        reason[decision==1] = "보정된 예측 집합에 불량만 포함"
        reason[valid & ~inside] = "학습 입력 범위 밖"
        reason[~valid] = "입력 누락·비유한 값·비양수 값 또는 잘못된 점수"
        return pd.DataFrame({"decision_code":decision,"decision":[OUTPUTS[int(d)] for d in decision],
                             "decision_reason":reason,"defect_score":p,
                             "possible_success":allow_success&valid,"possible_defect":allow_defect&valid})
