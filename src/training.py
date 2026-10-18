"""Optional same-recipe retraining to a new output, never update frozen models."""
import json
import pandas as pd
from .config import ROOT,FEATURES
from .dataset import read_inputs
from .daily_reference import daily_reference
from .predict import registry,load_model,LOADERS,writable_output
from .anomaly_models import LogisticAnomaly
from .dnn_anomaly import pseudo_labels
from .utils import save_json,write_csv


def train(name,seed,output):
    output=writable_output(output,'retraining')
    if output.exists(): raise FileExistsError('Choose a new output folder; frozen or previous fits are not overwritten')
    info=registry()['models'][name]; seed=seed if seed is not None else info['selected_seed']
    records,book=read_inputs(); train=records.loc[records.split.eq('train')]; val=records.loc[records.split.eq('validation')]
    if name=='Baseline':
        save_json(output/'metadata.json',{'model':'Baseline','trained':False,'all_outputs':'성공'})
        print('Baseline has no training'); return
    if name=='Logistic':
        model=LogisticAnomaly().fit(train); model.save(output); print(f'Logistic saved: {output}'); return
    saved=ROOT/f'results/models/{name}/seeds/seed{seed}/metadata.json'
    if not saved.exists(): raise ValueError('Available neural model seeds are 17,42,97')
    metadata=json.loads(saved.read_text(encoding='utf-8')); cfg=dict(metadata['config'])
    if name=='DNN':
        reference=load_model('Logistic'); model=LOADERS[name](**cfg)
        model.fit(train,pseudo_labels(reference,train[FEATURES]),val,pseudo_labels(reference,val[FEATURES]))
    else:
        inner_info=json.loads((ROOT/f'results/models/{name}/inner/seed{seed}/metadata.json').read_text(encoding='utf-8'))
        inner_cfg=dict(inner_info['config']); reference=daily_reference(records,book['result'])
        inner=train.loc[train.date.ne('2020-03-30')]; holdout=train.loc[train.date.eq('2020-03-30')]
        candidate=LOADERS[name](**inner_cfg).fit_counts(inner,reference,validation=holdout)
        candidate.save(output/'inner'); write_csv(pd.DataFrame(candidate.history),output/'inner_learning.csv')
        cfg['max_epochs']=candidate.validation_selection['joint_epoch']
        model=LOADERS[name](**cfg).fit_counts(train,reference)
        save_json(output/'selection_before_test.json',{'seed':seed,'inner_validation':candidate.validation_selection,'test_used':False,'calibration_used':False})
    model.save(output/'checkpoint'); write_csv(pd.DataFrame(model.history),output/'learning.csv')
    save_json(output/'retraining_manifest.json',{'model':name,'seed':seed,'source_checkpoint':info['model_folder'],'fit_dates':sorted(train.date.unique()),'test_used':False,'calibration_used':False,'frozen_registry_changed':False,'row_labels_are_inspection_truth':False})
    print(f'{name}, seed {seed}: 새 학습 저장 → {output}; 확정 모델은 변경하지 않음')
