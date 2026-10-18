import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.config import FEATURES,ROOT
from src.anomaly_models import AlwaysSuccess,LogisticAnomaly
from src.daily_reference import daily_reference


class AnomalyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(7)
        x=np.tile([.7,.7,5.,14.7,2.7,72.],(1000,1))
        x[:,2:]+=rng.normal(size=(1000,4))*[.3,.04,.02,.15]
        cls.frame=pd.DataFrame(x,columns=FEATURES)
        cls.model=LogisticAnomaly().fit(cls.frame)

    def test_baseline_needs_no_features_or_training(self):
        out=AlwaysSuccess().predict(pd.DataFrame({'record_id':['a','b']}))
        self.assertEqual(out.decision.tolist(),['성공','성공'])

    def test_center_far_and_boundary_have_three_outputs(self):
        model=self.model
        x=np.tile(model.center,(3,1))
        x[1,2]+=30*model.scale[2]
        # Find a point on the learned score=0.5 boundary, independent of labels.
        low,high=0.,30.
        for _ in range(60):
            mid=(low+high)/2
            point=model.center.copy(); point[2]+=mid*model.scale[2]
            if model.score(point[None,:])[0]<.5: low=mid
            else: high=mid
        x[2,2]+=(low+high)/2*model.scale[2]
        out=model.predict(pd.DataFrame(x,columns=FEATURES))
        self.assertEqual(out.decision.tolist(),['성공','불량','모르겠음'])

    def test_missing_values_and_new_thickness_abstain(self):
        x=self.frame.iloc[:3].copy()
        x.iloc[0,0]=.8; x.iloc[1,2]=np.nan; x.iloc[2,3]=-1
        self.assertEqual(self.model.predict(x).decision.tolist(),['모르겠음']*3)
        self.assertEqual(self.model.predict(x.drop(columns='voltage_v')).decision.tolist(),['모르겠음']*3)

    def test_model_is_train_only_and_replays(self):
        before=self.model.center.copy()
        self.model.predict(self.frame*2)
        np.testing.assert_array_equal(before,self.model.center)
        tmp=ROOT/'tmp'; tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='anomaly_test_',dir=tmp) as folder:
            self.model.save(folder)
            restored=LogisticAnomaly.load(folder)
            np.testing.assert_allclose(self.model.predict(self.frame).anomaly_score,
                                       restored.predict(self.frame).anomaly_score,atol=1e-12)

    def test_partial_or_missing_type_counts_are_not_zero_filled(self):
        records=pd.DataFrame({'date':['2020-01-01']*10+['2020-01-02']*10+['2020-01-03']*10,
                              'split':['train']*30})
        result=pd.DataFrame({'working time':['2020-01-01']*3+['2020-01-02'],
                             'defect type':[1,2,3,1],'defect':[2,1,1,2]})
        ref=daily_reference(records,result).set_index('date')
        self.assertEqual(ref.loc['2020-01-01','unique_defect_count_lower'],2)
        self.assertEqual(ref.loc['2020-01-01','unique_defect_count_upper'],4)
        self.assertTrue(pd.isna(ref.loc['2020-01-02','unique_defect_count_upper']))
        self.assertTrue(pd.isna(ref.loc['2020-01-03','unique_defect_count_lower']))


if __name__=='__main__': unittest.main()
