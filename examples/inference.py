"""Use the final resistance-five parallel DNN/GAMI checkpoint, seed 17."""
from functools import lru_cache
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.predict import load_model,predict


@lru_cache(maxsize=1)
def _model():
    return load_model('DNN_GAMI2')


def predict_batch(inputs):
    return predict(inputs,model=_model())


def predict_one(*,pressure_bar,current_ka,voltage_v,duration_ms):
    result=predict_batch({'pressure_bar':pressure_bar,'current_ka':current_ka,'voltage_v':voltage_v,'duration_ms':duration_ms}).iloc[0]
    score=float(result.anomaly_score); resistance=float(result.resistance_proxy_mohm)
    return {'classification':result.classification,'decision_code':int(result.decision_code),'anomaly_score':score if math.isfinite(score) else None,'resistance_proxy_mohm':resistance if math.isfinite(resistance) else None,'decision_reason':result.decision_reason}


def main():
    import argparse
    from src.predict import read_input,writable_output
    from src.utils import write_csv
    parser=argparse.ArgumentParser(description='최종 DNN_GAMI2 모델: 네 센서 입력 → 성공·실패·모르겠음')
    parser.add_argument('--input',type=Path,default=ROOT/'examples/four_sensor_weld.json')
    parser.add_argument('--output',type=Path,help='results/tables/predictions 아래의 새 CSV')
    args=parser.parse_args(); result=predict_batch(read_input(args.input))
    if args.output:
        output=writable_output(args.output)
        if output.exists(): raise FileExistsError('새 출력 파일 이름을 지정하세요.')
        write_csv(result,output)
        print(f'{len(result)}행 저장 → {output}')
    else: print(result.to_string(index=False))


if __name__=='__main__': main()
