"""Shared Final_Welding commands; frozen final checkpoint is the default."""
import argparse
import json
from pathlib import Path
from src.config import ROOT
from src.predict import registry,load_model,read_input,predict,writable_output
from src.utils import write_csv


def main():
    parser=argparse.ArgumentParser(description='센서 네 값 → 성공 / 실패 / 모르겠음; 저항 자동 계산')
    parser.add_argument('command',choices=['predict','compare','status','train'])
    parser.add_argument('--model',choices=list(registry()['models']),default='DNN_GAMI2')
    parser.add_argument('--seed',type=int,choices=[17,42,97])
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='predict':
        if args.input is None or args.output is None: parser.error('--input and --output are required')
        output=writable_output(args.output)
        if args.input.resolve()==output: parser.error('Input cannot be overwritten')
        model=load_model(args.model,args.seed); result=predict(read_input(args.input),model=model)
        write_csv(result,output); print(f'{args.model}: {len(result)}행 저장 → {output}')
    elif args.command=='compare':
        from src.evaluate import compare
        print(compare().query("split=='test'")[['model','success','failure','unknown','total_count_error','daily_count_mae']].to_string(index=False))
    elif args.command=='status':
        print(json.dumps(registry(),ensure_ascii=False,indent=2))
    elif args.command=='train':
        if args.output is None: parser.error('--output results/models/retrained/<new-name> is required')
        from src.training import train
        train(args.model,args.seed,writable_output(args.output,'retraining'))


if __name__=='__main__': main()
