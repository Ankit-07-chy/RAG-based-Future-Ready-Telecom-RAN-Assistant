"""
SimU5G simulation data parser.
Extracts failure scenarios and RCA patterns from SimU5G.
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s — %(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_SIMU5G_DIR = DATA_DIR / "raw" / "simu5g"
PROCESSED_DIR = DATA_DIR / "processed"

SIMU5G_JSON = RAW_SIMU5G_DIR / "simu5g_data.json"


def load_simu5g_data(json_path: Path) -> Dict[str, Any]:
    """Load SimU5G simulation data from JSON."""
    if not json_path.exists():
        logger.warning(f"SimU5G data not found: {json_path}")
        return {}

    logger.info(f"Loading SimU5G data from {json_path.name}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_simu5g_chunks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert SimU5G scenarios into searchable chunks.
    Each chunk represents a failure scenario with RCA and mitigation.
    """
    chunks = []

    # Extract scenarios
    scenarios = data.get('scenarios', [])
    logger.info(f"Processing {len(scenarios)} SimU5G scenarios...")

    for scenario_idx, scenario in enumerate(tqdm(scenarios, desc="Chunking scenarios")):
        # Build scenario chunk
        content_parts = [
            f"[SCENARIO] {scenario.get('scenario_name', 'Unknown')}",
            f"Failure Type: {scenario.get('failure_type', 'N/A')}",
            f"Simulation ID: {scenario.get('simulation_id', 'N/A')}",
        ]

        # Add scenario description
        desc = scenario.get('description', '')
        if desc:
            content_parts.append(f"\nDescription: {desc}")

        # Add failure details
        failure_details = scenario.get('failure_details', {})
        if failure_details:
            content_parts.append("\nFailure Details:")
            for key, value in failure_details.items():
                content_parts.append(f"  {key}: {value}")

        # Add RCA (Root Cause Analysis)
        rca = scenario.get('root_cause_analysis', {})
        if rca:
            content_parts.append("\nRoot Cause Analysis:")
            if 'causes' in rca:
                for cause in rca['causes']:
                    content_parts.append(f"  • {cause}")
            if 'contributing_factors' in rca:
                content_parts.append("Contributing Factors:")
                for factor in rca['contributing_factors']:
                    content_parts.append(f"    - {factor}")

        # Add mitigation steps
        mitigations = scenario.get('mitigation_steps', [])
        if mitigations:
            content_parts.append("\nMitigation Steps:")
            for i, mitigation in enumerate(mitigations, 1):
                content_parts.append(f"  {i}. {mitigation}")

        # Add optimization tips
        optimization = scenario.get('optimization_tips', [])
        if optimization:
            content_parts.append("\nOptimization Tips:")
            for tip in optimization:
                content_parts.append(f"  ✓ {tip}")

        # Add simulation parameters
        params = scenario.get('simulation_parameters', {})
        if params:
            content_parts.append("\nSimulation Parameters:")
            for param, value in params.items():
                content_parts.append(f"  {param}: {value}")

        content = "\n".join(content_parts)
        word_count = len(content.split())

        chunk = {
            'content': content,
            'source': 'simu5g',
            'scenario_id': scenario.get('id', f'scenario_{scenario_idx}'),
            'scenario_name': scenario.get('scenario_name', ''),
            'failure_type': scenario.get('failure_type', ''),
            'simulation_id': scenario.get('simulation_id', ''),
            'word_count': word_count,
            'chunk_type': 'simu5g_scenario'
        }
        chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks from SimU5G data")
    return chunks


def parse_simu5g_data(simu5g_dir: Path = None) -> List[Document]:
    """
    High-level wrapper: load Simu5G JSON from simu5g_dir and return LangChain Documents.
    Metadata keys: scenario_type, failure_mode, scenario_id, doc_type=simu5g.
    """
    if simu5g_dir is None:
        simu5g_dir = RAW_SIMU5G_DIR
    json_path = simu5g_dir / "simu5g_data.json"
    data = load_simu5g_data(json_path)
    if not data:
        logger.warning("No Simu5G data found — returning empty document list.")
        return []
    chunks = create_simu5g_chunks(data)
    docs: List[Document] = []
    for chunk in chunks:
        docs.append(Document(
            page_content=chunk["content"],
            metadata={
                "scenario_id":   chunk.get("scenario_id", "N/A"),
                "scenario_type": chunk.get("failure_type", chunk.get("scenario_name", "N/A")),
                "failure_mode":  chunk.get("failure_type", "N/A"),
                "simulation_id": chunk.get("simulation_id", "N/A"),
                "doc_type":      "simu5g",
                "source":        str(json_path),
                "word_count":    chunk.get("word_count", 0),
            },
        ))
    logger.info(f"parse_simu5g_data: returning {len(docs)} Documents")
    return docs


def save_simu5g_chunks(chunks: List[Dict[str, Any]], output_path: Path):
    """Save SimU5G chunks to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in tqdm(chunks, desc="Saving chunks"):
            f.write(json.dumps(chunk) + '\n')

    logger.info(f"✅ Saved {len(chunks)} SimU5G chunks to {output_path}")


def main():
    logger.info("="*70)
    logger.info("SIMU5G DATA PROCESSING")
    logger.info("="*70)

    # Load SimU5G data
    data = load_simu5g_data(SIMU5G_JSON)

    if not data:
        logger.warning("No SimU5G data found. Creating demo data...")
        # Create minimal demo
        data = {
            'scenarios': [
                {
                    'id': 'SIM-001',
                    'scenario_name': 'High Handover Failure in Dense Urban',
                    'failure_type': 'Handover Failure',
                    'simulation_id': 'SIM_DENSE_URBAN_001',
                    'description': 'UE experiencing excessive handover failures in dense urban environment',
                    'failure_details': {
                        'failure_rate': '12.5%',
                        'avg_ue_count': 500,
                        'environment': 'Dense Urban (Manhattan)'
                    },
                    'root_cause_analysis': {
                        'causes': [
                            'Measurement report latency',
                            'Mobility robustness optimization (MRO) misconfiguration'
                        ],
                        'contributing_factors': [
                            'High UE speed (vehicle)',
                            'Poor signal coverage transitions'
                        ]
                    },
                    'mitigation_steps': [
                        'Adjust handover margin and time-to-trigger parameters',
                        'Implement Idle Mode Signaling Reduction (IMSR)',
                        'Optimize cell rank configuration'
                    ],
                    'optimization_tips': [
                        'Use DL RS Power optimization',
                        'Implement Coordinated Multipoint (CoMP)',
                        'Enable Inter-RAT cell reselection'
                    ],
                    'simulation_parameters': {
                        'ue_speed': '60 km/h',
                        'carrier_frequency': '3.5 GHz',
                        'bandwidth': '100 MHz'
                    }
                }
            ]
        }

    # Create chunks
    chunks = create_simu5g_chunks(data)

    # Save
    output_path = PROCESSED_DIR / "simu5g_scenarios.jsonl"
    save_simu5g_chunks(chunks, output_path)

    logger.info("\n" + "="*70)
    logger.info(f"✅ SimU5G processing complete ({len(chunks)} chunks)")
    logger.info("="*70)


if __name__ == "__main__":
    main()
