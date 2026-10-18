"""Readable top-level commands sharing the preserved implementations."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ROOT,FEATURES
from .utils import write_csv,save_json


def _preserve_or_write(frame,path):
    """Keep byte-identical existing evidence when the recomputed table agrees."""
    if path.exists():
        old=pd.read_csv(path)
        pd.testing.assert_frame_equal(old,frame,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        return
    write_csv(frame,path)


def run_eda():
    parser=argparse.ArgumentParser(description='원본 전처리·시간순 분할·EDA 실행')
    parser.parse_args()
    from .dataset import read_inputs
    from .daily_reference import daily_reference
    frame,book=read_inputs()
    _preserve_or_write(frame,ROOT/'data/processed/weld_inputs.csv')
    stats=frame[FEATURES].agg(['min','max','nunique']).T.reset_index(names='feature')
    split=frame.groupby(['split','date'],as_index=False).agg(rows=('record_id','size'))
    daily=daily_reference(frame,book['result'])
    for table,path in [(stats,'results/eda/input_statistics.csv'),(split,'results/eda/date_splits.csv'),
                       (daily,'results/eda/daily_defect_reference.csv'),(daily,'results/tables/dataset_by_date.csv')]:
        _preserve_or_write(table,ROOT/path)
    summary={'rows':len(frame),'split_rows':frame.groupby('split').size().to_dict(),
             'source_workbook':str(ROOT/'data/raw/Welding Data Set_01.xlsx'),
             'existing_equal_tables_preserved':True,'individual_quality_labels_available':False}
    save_json(ROOT/'results/eda/eda_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def run_baseline():
    parser=argparse.ArgumentParser(description='Baseline 실행')
    parser.add_argument('--input',type=Path,default=ROOT/'data/processed/weld_inputs.csv')
    parser.add_argument('--output',type=Path,help='필요할 때 새 예측 CSV로 저장')
    args=parser.parse_args()
    from .predict import predict,read_input,writable_output
    frame=read_input(args.input);out=predict(frame,name='Baseline')
    if args.output:
        target=writable_output(args.output)
        if target.exists():raise FileExistsError('새 출력 파일 이름을 지정하세요.')
        write_csv(out,target)
    print(json.dumps({'model':'Baseline','rows':len(out),'decisions':out.classification.value_counts().to_dict()},ensure_ascii=False,indent=2))


def run_training(name):
    parser=argparse.ArgumentParser(description='기존 recipe로 새 모델을 학습하고 확정 모델은 유지')
    parser.add_argument('--seed',type=int,choices=[17,42,97])
    parser.add_argument('--output',type=Path,help='results/models/retrained 아래의 새 폴더')
    parser.add_argument('--dry-run',action='store_true',help='학습 계획만 표시')
    args=parser.parse_args()
    from .predict import registry,writable_output
    info=registry()['models'][name]
    seed=args.seed if args.seed is not None else info['selected_seed']
    output=args.output or ROOT/f'results/models/retrained/{name}_{datetime.now():%Y%m%d_%H%M%S}_seed{seed}'
    output=writable_output(output,'retraining')
    plan={'model':name,'seed':seed,'output':str(output),'frozen_checkpoint':info['model_folder'],
          'test_used_for_training':False,'calibration_used_for_training':False,'frozen_model_overwritten':False}
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    from .training import train
    train(name,seed,output)
