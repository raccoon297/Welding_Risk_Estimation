"""Build daily defect-count reference intervals from provided records."""
import numpy as np
import pandas as pd

def daily_reference(records,result):
    """Type counts constrain a daily union, but never identify failed welds."""
    ref=result.copy()
    ref['date']=pd.to_datetime(ref['working time']).dt.strftime('%Y-%m-%d')
    ref['type']=pd.to_numeric(ref['defect type'],errors='raise').astype(int)
    ref['count']=pd.to_numeric(ref['defect'],errors='raise')
    if ref.duplicated(['date','type']).any():
        raise ValueError('Duplicate date/type aggregate records require source review')
    rows=[]
    for date,group in records.groupby('date'):
        observed=ref.loc[ref.date.eq(date)].set_index('type')['count']
        if not observed.index.isin([1,2,3]).all() or (observed<0).any() or (observed>len(group)).any():
            raise ValueError('Invalid aggregate counts or unexpected defect type')
        complete=set(observed.index)=={1,2,3}
        rows.append({'date':date,'split':group.split.iloc[0],'production_rows':len(group),
                     **{f'reported_type_{i}':observed.get(i,np.nan) for i in [1,2,3]},
                     'all_three_types_reported':complete,
                     'reported_type_sum':observed.sum() if len(observed) else np.nan,
                     'unique_defect_count_lower':observed.max() if len(observed) else np.nan,
                     'unique_defect_count_upper':min(len(group),observed.sum()) if complete else np.nan})
    return pd.DataFrame(rows)
