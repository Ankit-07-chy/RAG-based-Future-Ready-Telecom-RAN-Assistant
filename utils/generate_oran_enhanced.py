import json
import random
from datetime import datetime, timedelta
from pathlib import Path
#--------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORAN_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "oran_datasets"
#--------------------------------------------------------------

def generate_enhanced_oran():
    alarms = [
        {
            "alarm_id": "ORAN-ALM-001",
            "alarm_name": "RU_FAULT",
            "perceived_severity": "CRITICAL",
            "probable_cause": "Radio Unit hardware failure",
            "specific_problem": "Power amplifier failure",
            "additional_text": "RU transmit power dropped below threshold"
        },
        {
            "alarm_id": "ORAN-ALM-002",
            "alarm_name": "CELL_OUTAGE",
            "perceived_severity": "CRITICAL",
            "probable_cause": "Cell configuration error",
            "specific_problem": "Cell barred due to inconsistent parameters",
            "additional_text": "gNB configuration mismatch detected"
        },
        {
            "alarm_id": "ORAN-ALM-003",
            "alarm_name": "HIGH_INTERFERENCE",
            "perceived_severity": "MAJOR",
            "probable_cause": "External interference detected",
            "specific_problem": "RSSI exceeding -85 dBm",
            "additional_text": "Interference in n78 band"
        },
        {
            "alarm_id": "ORAN-ALM-004",
            "alarm_name": "THROUGHPUT_DEGRADATION",
            "perceived_severity": "MAJOR",
            "probable_cause": "Resource congestion",
            "specific_problem": "DL throughput below 50% expected",
            "additional_text": "PRB utilization at 95%"
        },
        {
            "alarm_id": "ORAN-ALM-005",
            "alarm_name": "E2_LINK_FAILURE",
            "perceived_severity": "CRITICAL",
            "probable_cause": "E2 interface connection lost",
            "specific_problem": "SCTP heartbeat timeout",
            "additional_text": "Near-RT RIC unable to communicate with CU"
        },
        {
            "alarm_id": "ORAN-ALM-006",
            "alarm_name": "PRACH_FAILURE",
            "perceived_severity": "MAJOR",
            "probable_cause": "PRACH configuration issue",
            "specific_problem": "RACH success rate below 40%",
            "additional_text": "Preamble collision rate at 65%"
        },
        {
            "alarm_id": "ORAN-ALM-007",
            "alarm_name": "HANDOVER_FAILURE_RATE",
            "perceived_severity": "WARNING",
            "probable_cause": "Mobility parameter misconfiguration",
            "specific_problem": "HO failure rate exceeding 5%",
            "additional_text": "A3 event threshold needs adjustment"
        },
        {
            "alarm_id": "ORAN-ALM-008",
            "alarm_name": "F1_LINK_DEGRADATION",
            "perceived_severity": "WARNING",
            "probable_cause": "F1 interface packet loss",
            "specific_problem": "Packet loss > 1% on F1-U",
            "additional_text": "Potential midhaul congestion"
        }
    ]
    
    kpis = [
        {"name": "DL_PRB_Utilization", "range": [20, 80], "unit": "%"},
        {"name": "Avg_DL_UE_Throughput", "range": [50, 500], "unit": "Mbps"},
        {"name": "RRC_Connected_UEs", "range": [10, 500], "unit": "Count"},
        {"name": "CQI_Avg", "range": [5, 15], "unit": "Index"},
        {"name": "SSB_RSRP_Avg", "range": [-120, -70], "unit": "dBm"},
        {"name": "SSB_SINR_Avg", "range": [-10, 30], "unit": "dB"},
        {"name": "HO_Success_Rate", "range": [95, 100], "unit": "%"},
        {"name": "Call_Drop_Rate", "range": [0, 2], "unit": "%"},
        {"name": "RRC_Setup_Success", "range": [98, 100], "unit": "%"},
        {"name": "Avg_UL_Throughput", "range": [20, 100], "unit": "Mbps"},
    ]
    
    cells = [f"Cell_{i:03d}" for i in range(1, 11)]
    base_time = datetime.now()
    
    # Generate alarms
    alarm_instances = []
    for _ in range(100):
        alarm = random.choice(alarms)
        cell = random.choice(cells)
        timestamp = base_time - timedelta(
            days=random.randint(0, 7),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        alarm_instances.append({
            **alarm,
            "occurrence_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "cell_id": cell,
            "context": (
                f"O-RAN Alarm {alarm['alarm_id']}: {alarm['alarm_name']} "
                f"on {cell}. Severity: {alarm['perceived_severity']}. "
                f"Cause: {alarm['probable_cause']}. {alarm['specific_problem']}. "
                f"Details: {alarm['additional_text']}"
            )
        })
    
    # Generate KPIs
    kpi_records = []
    for cell in cells:
        for kpi in kpis:
            for hour in range(24):
                timestamp = base_time - timedelta(hours=hour)
                hour_factor = 1.0 + 0.3 * (1 - abs(12 - hour) / 12)
                
                if random.random() < 0.05:
                    value = kpi["range"][0] * random.uniform(0.1, 0.5)
                else:
                    value = random.uniform(kpi["range"][0], kpi["range"][1]) * hour_factor
                
                kpi_records.append({
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "cell_id": cell,
                    "kpi_name": kpi["name"],
                    "value": round(value, 2),
                    "unit": kpi["unit"],
                    "context": (
                        f"O-RAN KPI: {cell} - {kpi['name']} = {value:.2f} {kpi['unit']}"
                    )
                })
    
    dataset = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Synthetic O-RAN operational data for TelecomRAG",
            "total_alarms": len(alarm_instances),
            "total_kpi_records": len(kpi_records),
            "cells": len(cells),
            "time_range": "7 days"
        },
        "alarms": alarm_instances,
        "kpis": kpi_records
    }
    
    ORAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ORAN_OUTPUT_DIR / "oran_data.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated O-RAN dataset:")
    print(f"   - {len(alarm_instances)} alarm instances")
    print(f"   - {len(kpi_records)} KPI records")
    print(f"   - {len(cells)} cells")
    print(f"✅ Saved to: {output_path}")

if __name__ == "__main__":
    generate_enhanced_oran()
