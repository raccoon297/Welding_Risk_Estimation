"""Public four-sensor prediction API with fitted, optional material context."""
import numpy as np

from .config import FEATURES

SENSORS=FEATURES[2:]


def predict_weld(model,frame):
    context=frame.copy()
    fixed=getattr(model,'fixed_values',getattr(model,'center',None))
    if fixed is not None:
        for i,col in enumerate(FEATURES[:2]):
            if col not in context: context[col]=float(fixed[i])
    # Explicit thickness, including missing/invalid values, is never overwritten.
    out=model.predict(context)
    out['classification']=out.decision_code.map({0:'성공',1:'실패',-1:'모르겠음'})
    return out
