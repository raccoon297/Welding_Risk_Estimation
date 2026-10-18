"""DNN_GAMI2: five inputs including derived resistance, parallel GAMI/DNN."""
import json
from pathlib import Path

import numpy as np
import torch

from .config import FEATURES
from .resistance_inputs import ResistanceInputs,PROXIES
from .gami_components import ParallelGAMI
from .dnn_gami import DNNGAMI
from .utils import save_json,sha256


class DNNGAMI2(ResistanceInputs):
    display_name='DNN_GAMI2'
    mode='parallel_DNN_GAMI2_five_input_resistance'

    def __init__(self,**kwargs):
        super().__init__(feature='resistance',**kwargs)
        self.use_dnn=True

    def build_network(self):
        return ParallelGAMI(torch.zeros(self.anchor_count,int(self.active.sum())),
            self.main_hidden,self.pair_hidden,self.hidden,use_dnn=True)

    def _count_phase(self,name,limit,tx,hint,frame,reference,validation):
        self.network.dnn_enabled.fill_(name=='joint')
        # Same objective/optimizer/selection as DNN_GAMI; both branches see 5 inputs.
        return DNNGAMI._count_phase(self,name,limit,tx,hint,frame,reference,validation)

    def predict(self,frame):
        out=super().predict(frame)
        out['prediction_basis']='five_input_parallel_DNN_GAMI_aggregate_weak_supervision_not_inspection'
        return out

    def save(self,folder):
        super().save(folder)
        (Path(folder)/'resistance_inputs_weights.pt').rename(Path(folder)/'dnn_gami2_weights.pt')
        path=Path(folder)/'metadata.json'
        info=json.loads(path.read_text(encoding='utf-8'))
        info.update(model='DNN_GAMI2',use_dnn=True,DNN_parameters_present=True,
            logit_formula='bias + centered_main_sum + centered_pair_sum + sigmoid(alpha_logit) * centered_DNN',
            DNN_architecture=[5,*self.hidden,1],DNN_last_layer_initialization='zero',
            DNN_enabled_only_during_joint_training=True)
        save_json(path,info)

    @classmethod
    def load(cls,folder,allow_fixture=False):
        folder=Path(folder); info=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        if info['inference_mode']!=cls.mode or info['raw_feature_columns']!=FEATURES or not info['use_dnn']: raise ValueError('Hybrid mode/input mismatch')
        if info['feature']!='resistance' or info['proxy']!=PROXIES['resistance'] or info['model_feature_columns']!=[*FEATURES,PROXIES['resistance']['column']]: raise ValueError('Resistance definition mismatch')
        if info['training_label_provenance']=='synthetic_verification_only' and not allow_fixture: raise ValueError('Synthetic checkpoint rejected')
        weights=folder/'dnn_gami2_weights.pt'
        if sha256(weights)!=info['checkpoint_sha256']: raise ValueError('Hybrid checksum mismatch')
        model=cls(**info['config'])
        for key in ['mean','scale','fixed_values']: setattr(model,key,np.asarray(info[key],dtype=float))
        model.constant=np.asarray(info['constant'],dtype=bool); model.active=~model.constant
        for key in ['anchor_count','anchor_train_indices','policy','fit_rows','fit_dates','bag_audit','hint_statistics','stage_selection','validation_selection','pruning']: setattr(model,key,info[key])
        model.network=model.build_network(); model.network.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)); model.network.eval()
        model.all_rows_in_loss=True; model.provenance=info['training_label_provenance']
        return model


def predict_without_DNN(model,frame):
    """Ablation with all fitted parameters and the Train policy held fixed."""
    flag=bool(model.network.dnn_enabled)
    try:
        model.network.dnn_enabled.fill_(False)
        return model.predict(frame)
    finally:
        model.network.dnn_enabled.fill_(flag)
