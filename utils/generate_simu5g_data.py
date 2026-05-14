import json
import random
from datetime import datetime, timedelta

def generate_simu5g_scenarios():
    scenarios = [
        {
            "scenario_id": "SIM-001",
            "name": "PCI Collision",
            "description": "Two neighboring cells assigned same PCI causing interference",
            "symptoms": ["High interference", "Frequent HO failures", "Reduced SINR"],
            "affected_cells": ["Cell_001", "Cell_002"],
            "root_cause": "Incorrect PCI planning in dense urban deployment",
            "resolution": "Re-assign PCI for Cell_002 to avoid collision",
            "severity": "Major"
        },
        {
            "scenario_id": "SIM-002",
            "name": "RACH Overload",
            "description": "Excessive Random Access attempts causing preamble collision",
            "symptoms": ["RRC setup failures", "High PRACH utilization", "Access delays"],
            "affected_cells": ["Cell_005"],
            "root_cause": "Massive IoT device registration after power outage recovery",
            "resolution": "Adjust PRACH configuration (increase preambles, backoff timer)",
            "severity": "Critical"
        },
        {
            "scenario_id": "SIM-003",
            "name": "Beam Failure Recovery",
            "description": "UE unable to maintain beam alignment in mmWave frequency",
            "symptoms": ["RLF events", "Throughput drops", "Beam failure indications"],
            "affected_cells": ["Cell_008", "Cell_009"],
            "root_cause": "Moving obstacles in dense urban canyon affecting mmWave beams",
            "resolution": "Optimize beam management parameters",
            "severity": "Major"
        },
        {
            "scenario_id": "SIM-004",
            "name": "Handover Ping-Pong",
            "description": "Rapid successive handovers between two cells",
            "symptoms": ["High HO rate", "Signaling storm", "Poor user experience"],
            "affected_cells": ["Cell_003", "Cell_004"],
            "root_cause": "A3 event hysteresis configured too low (1dB)",
            "resolution": "Increase A3 hysteresis to 3dB, adjust TTT to 320ms",
            "severity": "Warning"
        },
        {
            "scenario_id": "SIM-005",
            "name": "Backhaul Congestion",
            "description": "Backhaul link saturated causing E2E throughput degradation",
            "symptoms": ["Low throughput", "High latency", "Packet drops"],
            "affected_cells": ["Cell_010", "Cell_011", "Cell_012"],
            "root_cause": "Unexpected traffic surge from stadium event",
            "resolution": "Implement QoS prioritization and traffic shaping",
            "severity": "Major"
        }
    ]
    
    base_time = datetime.now() - timedelta(days=7)
    for i, scenario in enumerate(scenarios):
        scenario["simulation_time"] = (base_time + timedelta(days=i)).strftime("%Y-%m-%d")
        scenario["context"] = (
            f"Simu5G Scenario {scenario['scenario_id']}: {scenario['name']}. "
            f"{scenario['description']}. Root Cause: {scenario['root_cause']}. "
            f"Resolution: {scenario['resolution']}. "
            f"Symptoms: {', '.join(scenario['symptoms'])}."
        )
    
    data = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Synthetic Simu5G failure scenarios for TelecomRAG RCA demos",
            "num_scenarios": len(scenarios)
        },
        "scenarios": scenarios
    }
    
    with open("data/simu5g/simu5g_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Generated Simu5G dataset:")
    print(f"   - {len(scenarios)} failure scenarios")
    print("✅ Saved to: data/simu5g/simu5g_data.json")

if __name__ == "__main__":
    generate_simu5g_scenarios()
