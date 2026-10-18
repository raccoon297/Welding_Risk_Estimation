from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data/raw/Welding Data Set_01.xlsx"
FEATURE_MAP = {
    "Thickness 1(mm)": "thickness_1_mm",
    "Thickness 2(mm)": "thickness_2_mm",
    "weld force(bar)": "pressure_bar",
    "weld current(kA)": "current_ka",
    "weld Voltage(v)": "voltage_v",
    "weld time(ms)": "duration_ms",
}
FEATURES = list(FEATURE_MAP.values())
OUTPUTS = {0: "성공", 1: "불량", -1: "모르겠음"}
RAW_ID_COLUMNS = {"working time": "date", "idx": "production_index",
                  "Machine_Name": "machine", "Item No": "item"}
LABEL_FIELDS = ["record_id", "workbook_sha256", "label", "is_verified", "label_source", "inspection_id"]


def directories():
    for relative in ["data/raw", "data/processed", "reports", "results/eda", "results/models",
                     "results/tables/experiments", "results/tables/final", "src/tests", "tmp"]:
        (ROOT/relative).mkdir(parents=True, exist_ok=True)
