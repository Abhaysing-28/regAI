"""
Synthetic data generator for RegAI project.
Generates fake regulatory report CSVs and pipeline logs.
No real/confidential data used.
"""
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import os

os.makedirs("data/reports", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

random.seed(42)
np.random.seed(42)

PIPELINE_STAGES = ["DATA_SOURCING", "SCHEMA_VALIDATION", "TRANSFORMATION", "RECONCILIATION", "OUTPUT_GENERATION"]

def generate_clean_report(n=200):
    return pd.DataFrame({
        "trade_id":        [f"TRD-{i:05d}" for i in range(n)],
        "report_date":     pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "counterparty_id": [f"CP-{random.randint(1000,9999)}" for _ in range(n)],
        "notional_amount": np.round(np.random.uniform(10_000, 5_000_000, n), 2),
        "currency":        np.random.choice(["USD", "EUR", "GBP", "JPY"], n),
        "status":          np.random.choice(["SETTLED", "PENDING", "FAILED"], n, p=[0.85, 0.1, 0.05]),
        "asset_class":     np.random.choice(["EQUITY", "BOND", "DERIVATIVE", "FX"], n),
        "jurisdiction":    np.random.choice(["EU", "UK", "US", "APAC"], n),
    })

def generate_anomalous_report(n=200):
    df = generate_clean_report(n)
    df.loc[10:15, "counterparty_id"] = None
    df.loc[30:32, "notional_amount"] = -99999.99
    df.loc[50, "currency"] = "XXX"
    df.loc[70, "trade_id"] = df.loc[0, "trade_id"]
    df.loc[71, "trade_id"] = df.loc[1, "trade_id"]
    df.loc[100, "notional_amount"] = 999_999_999_999.00
    df.loc[120:125, "jurisdiction"] = None
    return df

generate_clean_report().to_csv("data/reports/csdr_report_clean.csv", index=False)
generate_anomalous_report().to_csv("data/reports/csdr_report_anomalous.csv", index=False)

def generate_success_log():
    logs = []
    base_time = datetime(2024, 6, 15, 8, 0, 0)
    for i, stage in enumerate(PIPELINE_STAGES):
        logs.append({"timestamp": (base_time + timedelta(minutes=i*12)).isoformat(), "pipeline_id": "PIPE-CSDR-001", "stage": stage, "status": "SUCCESS", "records_in": 1000 - i*5, "records_out": 1000 - i*5, "duration_sec": random.randint(30, 120), "message": f"Stage {stage} completed successfully."})
    return logs

def generate_failure_log():
    logs = []
    base_time = datetime(2024, 6, 16, 8, 0, 0)
    for i, stage in enumerate(PIPELINE_STAGES):
        if stage == "TRANSFORMATION":
            logs.append({"timestamp": (base_time + timedelta(minutes=i*12)).isoformat(), "pipeline_id": "PIPE-CSDR-002", "stage": stage, "status": "FAILED", "records_in": 985, "records_out": 0, "duration_sec": 45, "error_code": "TRANSFORM_ERR_007", "message": "NullPointerException: Column counterparty_id contains unexpected null values at rows [10,11,12,13,14,15]. Downstream join operation aborted.", "stack_trace": "at TransformLayer.applyMappings(TransformLayer.py:142)"})
            break
        logs.append({"timestamp": (base_time + timedelta(minutes=i*12)).isoformat(), "pipeline_id": "PIPE-CSDR-002", "stage": stage, "status": "SUCCESS", "records_in": 1000-i*5, "records_out": 1000-i*5, "duration_sec": random.randint(30,120), "message": f"Stage {stage} completed successfully."})
    return logs

import json
with open("data/logs/pipeline_success.json", "w") as f:
    json.dump(generate_success_log(), f, indent=2)
with open("data/logs/pipeline_failure.json", "w") as f:
    json.dump(generate_failure_log(), f, indent=2)

print("Synthetic data generated successfully.")
