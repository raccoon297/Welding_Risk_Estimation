"""Portable five-model inference. Resistance is derived from the four sensors."""
import json
from pathlib import Path
import pandas as pd
from .config import ROOT
from .anomaly_models import AlwaysSuccess,LogisticAnomaly
from .dnn_anomaly import DNNAnomaly
from .dnn_gami import DNNGAMI
from .dnn_gami2 import DNNGAMI2
from .weld_input import predict_weld

LOADERS={'Logistic':LogisticAnomaly,'DNN':DNNAnomaly,'DNN_GAMI':DNNGAMI,'DNN_GAMI2':DNNGAMI2}


def registry():
    return json.loads((ROOT/'src/model_registry.json').read_text(encoding='utf-8'))


def load_model(name=None,seed=None):
    info=registry(); name=name or info['default_model']
    if name not in info['models']: raise ValueError('Unknown model: '+name)
    if name=='Baseline': return AlwaysSuccess()
    entry=info['models'][name]
    if seed is None: folder=ROOT/entry['model_folder']
    else:
        if name=='Logistic': raise ValueError('Logistic has one fitted checkpoint, no seed variants')
        if seed not in [17,42,97]: raise ValueError('Available seeds: 17,42,97')
        folder=ROOT/f'results/models/{name}/seeds/seed{seed}'
    return LOADERS[name].load(folder)


def read_input(path):
    path=Path(path)
    if path.suffix.lower()!='.json': return pd.read_csv(path)
    value=json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(value,dict): value=[value]
    if not isinstance(value,list) or not all(isinstance(row,dict) for row in value): raise ValueError('JSON object or object list required')
    rows=[]
    for item in value:
        item=dict(item)
        if 'thickness_mm' in item:
            thickness=item.pop('thickness_mm')
            if not isinstance(thickness,list) or len(thickness)!=2: raise ValueError('thickness_mm must have two values')
            item['thickness_1_mm'],item['thickness_2_mm']=thickness
        rows.append(item)
    return pd.DataFrame(rows)


def predict(frame,model=None,name=None):
    if isinstance(frame,dict): frame=pd.DataFrame([frame])
    elif not isinstance(frame,pd.DataFrame): frame=pd.DataFrame(frame)
    frame=frame.copy()
    if 'record_id' not in frame: frame['record_id']=[f'input-{i+1}' for i in range(len(frame))]
    result=predict_weld(model if model is not None else load_model(name),frame)
    # Public result wording is unified, while internal original model files stay intact.
    result['decision']=result['classification']
    return result


def writable_output(path,kind='predictions'):
    folders={'predictions':'results/tables/predictions','retraining':'results/models/retrained'}
    if kind not in folders: raise ValueError('Unknown output kind: '+kind)
    path=Path(path).resolve(); allowed=(ROOT/folders[kind]).resolve()
    if path==allowed or not path.is_relative_to(allowed): raise ValueError(f'Output must be inside {folders[kind]}/ to preserve frozen files')
    return path
