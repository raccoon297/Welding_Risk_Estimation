import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.config import ROOT,FEATURES
from src.dnn_anomaly import DNNAnomaly
from src.anomaly_metrics import proxy_metrics


class DNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(8)
        x=np.tile([.7,.7,5.,14.7,2.7,72.],(700,1))
        x[:,2]=rng.uniform(2,8,len(x))
        x[:,3:]+=rng.normal(size=(len(x),3))*[.03,.02,.1]
        frame=pd.DataFrame(x,columns=FEATURES)
        cls.train,cls.val=frame.iloc[:450].copy(),frame.iloc[450:].copy()
        cls.y=(cls.train.pressure_bar>5).astype(int).to_numpy()
        cls.vy=(cls.val.pressure_bar>5).astype(int).to_numpy()
        cls.model=DNNAnomaly(seed=17,max_epochs=180,patience=25).fit(
            cls.train,cls.y,cls.val,cls.vy,provenance='synthetic_verification_only')

    def query(self):
        x=np.tile(self.model.fixed_values,(3,1)); x[:,2]=[2.2,7.8,5.]
        return x

    def test_four_direct_sensor_inputs_affect_learned_scores(self):
        self.assertEqual(int(self.model.active.sum()),4)
        scores=self.model.score(self.query())
        self.assertGreater(scores[1]-scores[0],.6)

    def test_success_defect_and_boundary_abstention(self):
        x=self.query(); lo,hi=2.2,7.8
        for _ in range(45):
            mid=(lo+hi)/2; x[2,2]=mid
            if self.model.score(x[2:])[0]<.5: lo=mid
            else: hi=mid
        x[2,2]=(lo+hi)/2
        self.assertEqual(self.model.predict(pd.DataFrame(x,columns=FEATURES)).decision.tolist(),['성공','불량','모르겠음'])

    def test_invalid_and_unseen_thickness_abstain(self):
        x=pd.DataFrame(self.query(),columns=FEATURES)
        x.loc[0,'thickness_1_mm']=.8; x.loc[1,'voltage_v']=np.nan; x.loc[2,'duration_ms']=-1
        self.assertEqual(self.model.predict(x).decision.tolist(),['모르겠음']*3)
        self.assertEqual(self.model.predict(x.drop(columns='voltage_v')).decision.tolist(),['모르겠음']*3)

    def test_checkpoint_replay_provenance_and_integrity(self):
        tmp=ROOT/'tmp'; tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='dnn_test_',dir=tmp) as folder:
            self.model.save(folder)
            with self.assertRaises(ValueError): DNNAnomaly.load(folder)
            loaded=DNNAnomaly.load(folder,allow_fixture=True)
            np.testing.assert_allclose(self.model.score(self.val[FEATURES]),loaded.score(self.val[FEATURES]),atol=1e-12)
            path=Path(folder)/'dnn_weights.pt'; path.write_bytes(path.read_bytes()+b'tampered')
            with self.assertRaises(ValueError): DNNAnomaly.load(folder,allow_fixture=True)

    def test_fractional_labels_and_one_class_validation_rejected(self):
        y=self.y.astype(float); y[0]=.4
        with self.assertRaises(ValueError): DNNAnomaly(max_epochs=1).fit(self.train,y,self.val,self.vy)
        with self.assertRaises(ValueError): DNNAnomaly(max_epochs=1).fit(self.train,self.y,self.val,np.zeros(len(self.val)))

    def test_proxy_recall_keeps_abstained_anomalies(self):
        metrics=proxy_metrics([1,1,1,0,-1],[.9,.1,.5,.01,.8],[1,0,-1,0,1])
        self.assertEqual(metrics['pseudo_recall_including_abstentions'],1/3)
        self.assertEqual(metrics['pseudo_anomalies_abstained'],1)
        self.assertEqual(metrics['defect_candidate_outputs'],2)
        self.assertEqual(metrics['excluded_pseudo_boundary_rows'],1)


if __name__=='__main__': unittest.main()
