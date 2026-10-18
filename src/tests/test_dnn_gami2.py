import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.config import FEATURES,ROOT
from src.dnn_anomaly import seed_training
from src.resistance_inputs import augment_inputs
from src.dnn_gami import DNNGAMI
from src.dnn_gami2 import DNNGAMI2,predict_without_DNN


class HybridResistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(20); x=np.tile([.7,.7,2.34,14.73,2.702,72.],(300,1)); x[:,2:]+=rng.normal(size=(300,4))*[.2,.1,.025,.5]
        cls.frame=pd.DataFrame(x,columns=FEATURES); cls.frame['date']=np.repeat(['a','b','c'],100)
        cls.ref=pd.DataFrame({'date':['a','b','c'],'unique_defect_count_lower':[1.,2.,np.nan],
            'unique_defect_count_upper':[3.,4.,np.nan],'reported_type_sum':[3.,4.,np.nan]})
        args=dict(seed=17,main_epochs=10,pair_epochs=10,max_epochs=20,anchor_rows=64)
        cls.model=DNNGAMI2(**args).fit_counts(cls.frame,cls.ref,provenance='synthetic_verification_only')

    def test_both_branches_see_five_inputs(self):
        m=self.model; self.assertEqual(int(m.active.sum()),5); self.assertEqual(len(m.network.main),5)
        self.assertEqual(m.network.dnn[0].in_features,5); self.assertTrue(m.network.dnn_enabled)
        self.assertTrue(m.use_dnn); self.assertEqual(len(m.network.interactions),10)
        self.assertGreater(np.sqrt(np.mean(m.explain(self.frame[FEATURES])['dnn']**2)),0)

    def test_zero_initial_DNN_output(self):
        seed_training(42)
        m=DNNGAMI2();m.active=np.array([False,False,True,True,True,True,True]);m.anchor_count=64
        network=m.build_network();network.dnn_enabled.fill_(True)
        self.assertTrue(torch.equal(network.components(torch.ones(10,5))['dnn'],torch.zeros(10)))

    def test_original_sensor_hint_and_raw_resistance_formula(self):
        from src.dnn_gami import distance_hint
        _,hint=distance_hint(self.frame[FEATURES].to_numpy())
        self.assertEqual(self.model.hint_statistics,hint)
        x=augment_inputs(self.frame[FEATURES].to_numpy(),'resistance')
        np.testing.assert_array_equal(x[:,-1],self.frame.voltage_v/self.frame.current_ka)

    def test_DNN_ablation_is_exact_and_restores_parameters_and_flag(self):
        m=self.model; states={k:v.clone() for k,v in m.network.state_dict().items()}; policy=dict(m.policy)
        c=m.explain(self.frame[FEATURES]); expected=1/(1+np.exp(-(c['logit']-c['dnn'])))
        ablated=predict_without_DNN(m,self.frame)
        np.testing.assert_allclose(ablated.anomaly_score,expected,atol=2e-6)
        self.assertEqual(m.policy,policy)
        for k,v in states.items(): self.assertTrue(torch.equal(v,m.network.state_dict()[k]))

    def test_four_inputs_formula_and_invalid_guards(self):
        m=self.model; np.testing.assert_array_equal(m.predict(self.frame[FEATURES[2:]]).anomaly_score,m.predict(self.frame).anomaly_score)
        rows=self.frame[FEATURES].iloc[:3].copy(); rows.iloc[0,3]=0; rows.iloc[1,4]=np.nan; rows.iloc[2,0]=.8
        self.assertEqual(m.predict(rows).classification.tolist(),['모르겠음']*3)

    def test_checkpoint_replay_mode_and_integrity(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            p=Path(temp); self.model.save(p)
            with self.assertRaises(ValueError): DNNGAMI2.load(p)
            restored=DNNGAMI2.load(p,allow_fixture=True)
            np.testing.assert_array_equal(restored.predict(self.frame).anomaly_score,self.model.predict(self.frame).anomaly_score)
            with (p/'dnn_gami2_weights.pt').open('ab') as f: f.write(b'tamper')
            with self.assertRaises(ValueError): DNNGAMI2.load(p,allow_fixture=True)
