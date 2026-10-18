from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.config import ROOT,FEATURES
from src.dnn_gami import DNNGAMI,interval_loss,make_bags,fit_count_policy,apply_policy


class CountTests(unittest.TestCase):
    def test_count_gradient_includes_each_bag_row_without_row_labels(self):
        p=torch.tensor([.1,.2,.15,.05],requires_grad=True)
        loss=interval_loss(p,[(torch.arange(4),1.,2.)]); loss.backward()
        self.assertTrue((p.grad<0).all())
        self.assertEqual(len(p.grad),4)

    def test_interval_zero_inside_and_partial_upper_not_fabricated(self):
        p=torch.tensor([.6,.7])
        self.assertEqual(float(interval_loss(p,[(torch.arange(2),1.,2.)])),0.)
        self.assertEqual(float(interval_loss(p,[(torch.arange(2),1.,None)])),0.)

    def test_missing_date_skipped_and_bad_interval_rejected(self):
        frame=pd.DataFrame({'date':['a','a','b','b']})
        ref=pd.DataFrame({'date':['a','b'],'unique_defect_count_lower':[1.,np.nan],'unique_defect_count_upper':[2.,np.nan]})
        bags,audit=make_bags(frame,ref)
        self.assertEqual([r['date'] for r in audit],['a'])
        self.assertEqual(len(bags),1)
        ref.loc[0,'unique_defect_count_upper']=3
        with self.assertRaises(ValueError): make_bags(frame,ref)

    def test_global_policy_preserves_score_ties_and_limits_training_abstention(self):
        scores=np.r_[np.linspace(.01,.4,96),[.8,.8,.9,.9]]
        dates=np.array(['a']*100)
        ref=pd.DataFrame({'date':['a'],'unique_defect_count_lower':[2.],'unique_defect_count_upper':[4.],'reported_type_sum':[4.]})
        policy=fit_count_policy(scores,dates,ref,.01)
        codes=apply_policy(scores,policy)
        self.assertEqual(int((codes==1).sum()),4)
        self.assertLessEqual(int((codes==-1).sum()),1)
        self.assertEqual(codes[96],codes[97])
        self.assertFalse(policy['inference_uses_date_or_count'])

    def test_constant_score_cannot_be_split_to_force_daily_quota(self):
        p=np.full(100,.03); ref=pd.DataFrame({'date':['a'],'unique_defect_count_lower':[2.],'unique_defect_count_upper':[4.],'reported_type_sum':[4.]})
        codes=apply_policy(p,fit_count_policy(p,['a']*100,ref))
        self.assertEqual(len(np.unique(codes)),1)
        self.assertNotEqual(int((codes==1).sum()),4)


class DNNGAMITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(25); x=np.tile([.7,.7,2.3,14.7,2.7,72.],(360,1))
        x[:,2:]+=rng.normal(size=(360,4))*[.15,.06,.015,.3]
        cls.frame=pd.DataFrame(x,columns=FEATURES); cls.frame['date']=np.repeat(['a','b','c'],120)
        cls.reference=pd.DataFrame({'date':['a','b','c'],'unique_defect_count_lower':[1.,2.,np.nan],
                                   'unique_defect_count_upper':[3.,4.,np.nan],'reported_type_sum':[3.,4.,np.nan]})
        cls.model=DNNGAMI(seed=17,main_epochs=10,pair_epochs=10,max_epochs=20,anchor_rows=64).fit_counts(
            cls.frame,cls.reference,provenance='synthetic_verification_only')

    def test_all_rows_participate_and_no_hard_row_labels(self):
        self.assertEqual(self.model.fit_rows,360)
        self.assertTrue(self.model.all_rows_in_loss)
        self.assertEqual(sum(b['rows'] for b in self.model.bag_audit),240)
        self.assertEqual(self.model.pseudo_support['soft_hint_rows'],360)
        self.assertEqual(self.model.pseudo_support['excluded_boundary'],0)

    def test_checkpoint_replay_integrity_and_fixture_protection(self):
        temp=ROOT/'tmp'; temp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='DNN_GAMI_test_',dir=temp) as folder:
            self.model.save(folder)
            with self.assertRaises(ValueError): DNNGAMI.load(folder)
            restored=DNNGAMI.load(folder,allow_fixture=True)
            np.testing.assert_allclose(restored.score(self.frame[FEATURES]),self.model.score(self.frame[FEATURES]),atol=1e-12)
            self.assertEqual(restored.predict(self.frame).decision.tolist(),self.model.predict(self.frame).decision.tolist())
            path=Path(folder)/'dnn_gami_weights.pt'; path.write_bytes(path.read_bytes()+b'tampered')
            with self.assertRaises(ValueError): DNNGAMI.load(folder,allow_fixture=True)

    def test_invalid_and_new_thickness_abstain_and_equal_inputs_equal_output(self):
        frame=self.frame.iloc[:3].copy(); frame.iloc[1]=frame.iloc[0]
        out=self.model.predict(frame)
        self.assertEqual(out.anomaly_score.iloc[0],out.anomaly_score.iloc[1])
        frame.loc[frame.index[0],'thickness_1_mm']=.8
        frame.loc[frame.index[1],'current_ka']=np.nan
        frame.loc[frame.index[2],'duration_ms']=-1
        self.assertEqual(self.model.predict(frame).decision.tolist(),['모르겠음']*3)

    def test_three_decision_policy_and_uncalibrated_score(self):
        policy={'success_below':.3,'defect_at_or_above':.7}
        np.testing.assert_array_equal(apply_policy([.1,.5,.9,np.nan],policy),[0,-1,1,-1])
        self.assertFalse(self.model.policy['is_quality_probability_calibration'])


if __name__=='__main__': unittest.main()
