"""Audited per-weld inputs; verified inspection labels must join by record_id.

Missing inspection labels are never filled from daily counts, row position,
anomaly scores, model predictions, or an assumed failure rate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .utils import sha256, save_json, write_csv
from .config import ROOT, WORKBOOK, FEATURES, FEATURE_MAP, RAW_ID_COLUMNS, LABEL_FIELDS, directories


def read_inputs():
    book = pd.read_excel(WORKBOOK, sheet_name=None)
    raw = book["Raw data"].rename(columns={**FEATURE_MAP, **RAW_ID_COLUMNS}).copy()
    raw.insert(0,"source_excel_row",np.arange(2,len(raw)+2))
    raw.insert(0,"record_id",[f"weld-{i:06d}" for i in range(2,len(raw)+2)])
    raw["date"] = pd.to_datetime(raw.date,errors="raise").dt.strftime("%Y-%m-%d")
    raw["workbook_sha256"] = sha256(WORKBOOK)
    for col in FEATURES:
        raw[col] = pd.to_numeric(raw[col],errors="raise")
    if not np.isfinite(raw[FEATURES].to_numpy()).all() or (raw[FEATURES]<=0).any().any():
        raise ValueError("원본 입력에 결측·무한값 또는 비양수 값이 있습니다.")
    dates = sorted(raw.date.unique())
    if len(dates)<6:
        raise ValueError("시간순 train/validation/calibration/test 분할에 날짜가 부족합니다.")
    split = {d:"train" for d in dates[:-4]}
    split[dates[-4]],split[dates[-3]] = "validation","calibration"
    split.update({d:"test" for d in dates[-2:]})
    raw["split"] = raw.date.map(split)
    return raw,book


def validate_labels(records,labels):
    missing = set(LABEL_FIELDS)-set(labels.columns)
    if missing:
        raise ValueError(f"검사 정답 열 누락: {sorted(missing)}")
    if labels.record_id.duplicated().any():
        raise ValueError("검사 정답에 중복 record_id가 있습니다.")
    if not set(labels.record_id).issubset(set(records.record_id)):
        raise ValueError("원본 입력에 없는 record_id가 있습니다.")
    verified = labels.is_verified.astype(str).str.lower().isin(["true","1"])
    has_label = labels.label.notna() & labels.label.astype(str).str.strip().ne("")
    if (has_label & ~verified).any():
        raise ValueError("추정 또는 미검증 라벨을 실제 검사 정답으로 사용할 수 없습니다.")
    selected = labels.loc[verified & has_label].copy()
    selected["label"] = pd.to_numeric(selected.label,errors="raise")
    if not selected.label.isin([0,1]).all():
        raise ValueError("품질 정답은 0=성공, 1=불량만 허용합니다. 모르겠음은 정답 클래스가 아닙니다.")
    if not selected.workbook_sha256.eq(sha256(WORKBOOK)).all():
        raise ValueError("검사 정답의 원본 파일 해시가 다릅니다.")
    for field in ["label_source","inspection_id"]:
        if selected[field].isna().any() or selected[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"실제 검사 근거 필요: {field}")
    selected["label"] = selected.label.astype(int)
    return selected[LABEL_FIELDS]


def training_records():
    inputs = pd.read_csv(ROOT/"data/processed/weld_inputs.csv")
    path = ROOT/"data/processed/verified_quality_labels.csv"
    if not path.is_file():
        raise ValueError("개별 용접의 실제 검사 정답이 없습니다. 집계 건수를 행별 정답으로 바꾸지 않습니다.")
    labels = validate_labels(inputs,pd.read_csv(path))
    if labels.empty:
        raise ValueError("검증된 개별 용접 정답은 0개입니다.")
    joined = inputs.merge(labels.drop(columns="workbook_sha256"),on="record_id",validate="one_to_one")
    for split in ["train","validation","calibration"]:
        frame = joined.loc[joined.split.eq(split)]
        if set(frame.label)!= {0,1}:
            raise ValueError(f"{split} 구간에 실제 성공·불량 정답이 모두 필요합니다. 현재 {len(frame)}행.")
    return joined


def import_labels(path):
    directories()
    inputs = pd.read_csv(ROOT/"data/processed/weld_inputs.csv")
    selected = validate_labels(inputs,pd.read_csv(path))
    if selected.empty:
        raise ValueError("가져올 검증된 검사 정답이 없습니다.")
    destination = ROOT/"data/processed/verified_quality_labels.csv"
    if destination.exists():
        previous = pd.read_csv(destination)
        if set(previous.record_id)&set(selected.record_id):
            raise ValueError("이미 등록된 record_id입니다. 검사 정답을 자동 덮어쓰지 않습니다.")
        selected = pd.concat([previous,selected],ignore_index=True)
    digest = sha256(path)
    snapshot = ROOT/f"data/raw/quality_labels_{digest}.csv"
    if not snapshot.exists():
        snapshot.write_bytes(Path(path).read_bytes())
    write_csv(selected,destination)
    save_json(ROOT/"results/label_import.json",{"source_file_sha256":digest,"verified_rows":len(selected),
        "quality_classes":{"0":"성공","1":"불량"},"daily_counts_used_as_row_labels":False})
    return len(selected)


def audit_stage1():
    directories()
    records,book = read_inputs()
    write_csv(records,ROOT/"data/processed/weld_inputs.csv")
    # A review form, never a labelled training dataset.
    template = records[["record_id","source_excel_row","date","production_index","workbook_sha256",*FEATURES]].copy()
    for field in ["label","is_verified","label_source","inspection_id"]:
        template[field] = ""
    template_path = ROOT/"data/processed/inspection_label_template.csv"
    if not template_path.exists():
        write_csv(template,template_path)
    result = book["result"]
    stats = [{"feature":f,"min":float(records[f].min()),"max":float(records[f].max()),
              "unique_values":int(records[f].nunique()),"missing":int(records[f].isna().sum())} for f in FEATURES]
    write_csv(pd.DataFrame(stats),ROOT/"results/eda/input_statistics.csv")
    write_csv(records.groupby(["split","date"],as_index=False).agg(rows=("record_id","size")),
              ROOT/"results/eda/date_splits.csv")
    wb = load_workbook(WORKBOOK,read_only=True,data_only=True)
    mapping = {str(wb["result"][f"F{i}"].value):wb["result"][f"G{i}"].value for i in [2,3,4]}
    wb.close()
    supplied_row_labels = set(result.columns)&{"record_id","quality_label","label","pass_fail"}
    label_file = ROOT/"data/processed/verified_quality_labels.csv"
    verified_count = len(validate_labels(records,pd.read_csv(label_file))) if label_file.exists() else 0
    status = {"objective":"single_weld_success_defect_abstention","source_sha256":sha256(WORKBOOK),
        "rows":len(records),"input_categories":5,"numeric_features":FEATURES,
        "thickness_handling":"two sheet thicknesses preserved separately; no unsupported sum assumption",
        "fixed_features":[s["feature"] for s in stats if s["unique_values"]==1],
        "source_sheets":list(book),"result_columns":list(map(str,result.columns)),
        "defect_type_names_from_F2_G2_to_F4_G4":mapping,
        "result_is_per_weld_ground_truth":False,"result_has_row_label_columns":bool(supplied_row_labels),
        "verified_quality_label_rows":verified_count,"unlabelled_rows":len(records)-verified_count,
        "unknown_is_true_quality_class":False,"unknown_means_abstention":True,
        "row_quality_labels_fabricated":False,"labels_created_from_counts":False,
        "prediction_model_trained":False,"stage1_status":"input_audit_complete_label_join_unresolved",
        "historical_sensor_data_viewed":True,"classification_quality_performance_verified":False}
    save_json(ROOT/"results/stage1_audit.json",status)
    write_stage1_report(status,stats)
    print(json.dumps(status,ensure_ascii=False,indent=2))
    return status


def write_stage1_report(status,stats):
    text = "# 1단계 재시작: 개별 용접 분류용 데이터 감사\n\n"
    text += "목표는 두께·가압력·전류·전압·시간 한 세트를 입력받아 성공/불량/모르겠음을 반환하는 분류기다. 모르겠음은 판정 보류이며 실제 품질의 세 번째 클래스가 아니다.\n\n"
    text += f"원본 {status['rows']:,}행을 보존했다. 원본의 두 소재 두께를 각각 보존하여 5가지 입력 종류를 수치 열 6개로 표현한다. 두 두께 모두 0.7 mm이므로 다른 두께의 판정은 현 데이터로 학습하지 못한다.\n\n"
    text += "| 입력 열 | 최소 | 최대 | 고유값 |\n|---|---:|---:|---:|\n"
    for s in stats:
        text += f"| {s['feature']} | {s['min']:g} | {s['max']:g} | {s['unique_values']} |\n"
    text += f"\n현재 실제 검사와 연결된 개별 정답은 {status['verified_quality_label_rows']}개다. result의 idx는 검사 집계의 행 번호이며 Raw data의 생산순번과 임의 조인하지 않는다. 날짜별·유형별 건수에서 실패 행을 임의 선택하지 않는다.\n\n"
    text += "검사 정답 양식은 data/processed/inspection_label_template.csv다. label=0은 성공, 1은 불량이며 미검사는 빈칸으로 유지한다. 검사 근거 label_source와 inspection_id, 원본 workbook_sha256 및 record_id를 함께 등록해야 한다.\n\n"
    text += "입력 데이터 감사는 완료했지만 실제 정답 연결은 미완료다. 이 상태를 분류 모델 학습·성능 검증 완료로 표시하지 않는다. 기존 건수 MAE 순위는 새 분류 순위가 아니다.\n"
    (ROOT/"reports/stage1_restart.md").write_text(text,encoding="utf-8")
