"""Compare frozen models against the supplied daily counts without storing prediction CSVs."""
import pandas as pd
from .dataset import read_inputs
from .daily_reference import daily_reference
from .predict import registry, load_model, predict
from .count_evaluation import count_alignment


def compare():
    data, workbook = read_inputs()
    reference = daily_reference(data, workbook['result'])
    rows = []
    for name, info in registry()['models'].items():
        output = predict(data, model=load_model(name))
        for split in ('train', 'validation', 'calibration', 'test'):
            mask = data['split'].eq(split).to_numpy()
            rows.append({'model': name, 'model_label': info['label'], 'split': split,
                         **count_alignment(output.loc[mask, 'decision_code'], data.loc[mask, 'date'], reference)})
    return pd.DataFrame(rows)
