"""Count agreement against supplied type sums, not an all-success target."""
import numpy as np
import pandas as pd


def complete_count_dates(reference,dates):
    return reference.loc[reference.date.isin(dates)&reference.all_three_types_reported.eq(True)&reference.reported_type_sum.notna()]


def count_alignment(codes,dates,reference):
    codes=np.asarray(codes); dates=np.asarray(dates)
    if codes.shape!=dates.shape or not np.isin(codes,[-1,0,1]).all(): raise ValueError('Aligned three decisions required')
    observed=complete_count_dates(reference,dates)
    actual=[]; target=[]
    for r in observed.itertuples():
        actual.append(int((codes[dates==r.date]==1).sum())); target.append(float(r.reported_type_sum))
    return {'complete_reported_days':len(actual),'target_defect_total':float(sum(target)) if actual else None,
            'predicted_defect_total_complete_days':int(sum(actual)) if actual else None,
            'total_count_error':float(abs(sum(actual)-sum(target))) if actual else None,
            'daily_count_mae':float(np.mean(abs(np.asarray(actual)-target))) if actual else None,
            'success':int((codes==0).sum()),'failure':int((codes==1).sum()),'unknown':int((codes==-1).sum()),
            'rows':len(codes),'unknown_rate':float((codes==-1).mean()) if len(codes) else None}


def fit_reconstruction_policy(scores,dates,reference,unknown_budget=.01):
    """Global score cut, train counts only, with a narrow bounded unknown band."""
    scores=np.asarray(scores,dtype=float); dates=np.asarray(dates)
    if scores.ndim!=1 or scores.shape!=dates.shape or not len(scores) or not np.isfinite(scores).all() or (scores<0).any() or not 0<=unknown_budget<1:
        raise ValueError('Finite nonnegative aligned reconstruction errors required')
    ref=complete_count_dates(reference,dates)
    if not len(ref): raise ValueError('Complete training count days required')
    values=np.unique(scores)
    cuts=np.r_[np.nextafter(values[0],-np.inf),values[:-1]+(values[1:]-values[:-1])/2,np.nextafter(values[-1],np.inf)]
    counts=[]; targets=[]
    for r in ref.itertuples():
        ordered=np.sort(scores[dates==r.date]); counts.append(len(ordered)-np.searchsorted(ordered,cuts,side='left')); targets.append(r.reported_type_sum)
    counts=np.asarray(counts); targets=np.asarray(targets)
    total=abs(counts.sum(0)-targets.sum()); daily=abs(counts-targets[:,None]).mean(0)
    index=min(range(len(cuts)),key=lambda k:(total[k],daily[k],-cuts[k]))
    upper=float(cuts[index]); lower=upper if len(values)==1 or upper<=0 else upper*.95
    allowed=int(len(scores)*unknown_budget); ordered=np.sort(scores); at=np.searchsorted(ordered,upper,side='left')
    if int(((scores>=lower)&(scores<upper)).sum())>allowed:
        lower=min(upper,float(np.nextafter(ordered[max(0,at-allowed-1)],np.inf))) if at else upper
    return {'success_below':lower,'failure_at_or_above':upper,'unknown_budget':unknown_budget,
            'training_unknown_rate':float(((scores>=lower)&(scores<upper)).mean()),
            'training_total_count_error':float(total[index]),'training_daily_count_mae':float(daily[index]),
            'fit_dates':sorted(ref.date.unique()),'target':'supplied_complete_daily_type_sum_reference',
            'selection_order':['total_count_error','daily_count_mae','higher_cut'],
            'score_is_quality_probability':False,'inference_needs_date_or_count':False}


def reconstruction_decisions(scores,policy):
    p=np.asarray(scores,dtype=float); d=np.full(len(p),-1,dtype=int)
    valid=np.isfinite(p)&(p>=0)
    d[valid&(p<policy['success_below'])]=0; d[valid&(p>=policy['failure_at_or_above'])]=1
    return d
