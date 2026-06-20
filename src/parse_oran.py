"""
ORAN dataset Parser and chunker.
Fixed version with proper field mapping, no data loss, and streaming support.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Generator, Tuple
import logging
from tqdm import tqdm
from datetime import datetime
from langchain_core.documents import Document
import gc
#----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_ORAN_DIR = DATA_DIR / "raw" / "oran_datasets"
PROCESSED_DIR = DATA_DIR / "processed"
ORAN_JSON = RAW_ORAN_DIR / "oran_data.json"
logs_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(logs_dir, exist_ok=True)
#----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, f"oran_parser_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
#----------------------------------------------------

def load_oran_data(json_path: Path) -> Dict[str, Any]:
    """Load O-RAN data from JSON."""
    if not json_path.exists():
        logger.error(f"O-RAN data not found: {json_path}")
        return {}

    logger.info(f"Loading O-RAN data from {json_path.name}.")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
#----------------------------------------------------

def create_oran_chunks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert O-RAN alarms/KPIs into searchable chunks (FIXED VERSION).
    Correctly maps actual data fields and processes all KPI records.
    """
    chunks = []

    # Extract alarms section
    alarms = data.get('alarms', [])
    logger.info(f"Processing {len(alarms)} alarms.")

    for alarm_idx, alarm in enumerate(tqdm(alarms, desc="Chunking alarms")):
        # FIXED: Use correct field names from actual data
        alarm_id = alarm.get('alarm_id', f"alarm_{alarm_idx}")
        alarm_name = alarm.get('alarm_name', '')
        cell_id = alarm.get('cell_id', 'N/A')
        severity = alarm.get('perceived_severity', 'unknown')
        timestamp = alarm.get('occurrence_time', '')
        probable_cause = alarm.get('probable_cause', '')
        specific_problem = alarm.get('specific_problem', '')
        additional_text = alarm.get('additional_text', '')
        context = alarm.get('context', '')

        # Build alarm content with all available fields
        content_parts = [
            f"[ALARM] {alarm_id} - {alarm_name}",
            f"Cell ID: {cell_id}",
            f"Severity: {severity}",
            f"Timestamp: {timestamp}",
            f"Probable Cause: {probable_cause}",
            f"Specific Problem: {specific_problem}",
            f"Additional Info: {additional_text}",
            f"Full Context: {context}"
        ]
        content = "\n".join(content_parts)
        word_count = len(content.split())

        chunk = {
            'content': content,
            'source': 'oran',
            'alarm_id': alarm_id,  # FIXED: Use alarm_id directly
            'alarm_name': alarm_name,
            'alarm_type': alarm.get('alarm_type', ''),
            'cell_id': cell_id,
            'severity': severity,
            'probable_cause': probable_cause,
            'timestamp': timestamp,
            'word_count': word_count,
            'chunk_type': 'oran_alarm'
        }
        chunks.append(chunk)

    logger.info(f"Created {len(chunks)} alarm chunks")

    # Extract KPI patterns - FIXED: Process ALL records, not just 100
    kpi_data = data.get('kpis', [])
    logger.info(f"Processing {len(kpi_data)} KPI records (ALL records, no truncation)...")

    for kpi_idx, kpi in enumerate(tqdm(kpi_data, desc="Chunking KPIs")):  # FIXED: Removed [:100] limit
        # FIXED: Use correct field names from actual data structure
        kpi_name = kpi.get('kpi_name', 'Unknown')
        cell_id = kpi.get('cell_id', 'N/A')
        timestamp = kpi.get('timestamp', 'N/A')
        value = kpi.get('value', 'N/A')
        unit = kpi.get('unit', '')  # FIXED: Capture unit field
        context = kpi.get('context', '')

        # Build KPI chunk with actual data structure
        content_parts = [
            f"[KPI Report] {kpi_name}",
            f"Cell: {cell_id}",
            f"Period: {timestamp}",
            f"Value: {value} {unit}",
            f"Context: {context}"
        ]

        content = "\n".join(content_parts)
        word_count = len(content.split())

        chunk = {
            'content': content,
            'source': 'oran',
            'kpi_id': kpi.get('id', f'kpi_{kpi_idx}'),
            'kpi_name': kpi_name,  # FIXED: Use kpi_name
            'cell_id': cell_id,
            'value': value,
            'unit': unit,  # FIXED: Include unit
            'timestamp': timestamp,
            'word_count': word_count,
            'chunk_type': 'oran_kpi'  # FIXED: Correct chunk type
        }
        chunks.append(chunk)

    logger.info(f"Created {len(kpi_data)} KPI chunks (TOTAL: {len(chunks)} chunks)")
    return chunks


def parse_oran_data(oran_dir: Path = None) -> List[Document]:
    """
    High-level wrapper: load O-RAN JSON and return LangChain Documents.
    Fixed version with correct field mapping.
    """
    if oran_dir is None:
        oran_dir = RAW_ORAN_DIR
    json_path = oran_dir / "oran_data.json"
    data = load_oran_data(json_path)
    if not data:
        logger.warning("No O-RAN data found — returning empty document list.")
        return []

    chunks = create_oran_chunks(data)
    docs: List[Document] = []

    for chunk in chunks:
        # FIXED: Use correct doc_type based on chunk_type
        doc_type = chunk.get('chunk_type', 'oran_unknown')

        docs.append(Document(
            page_content=chunk["content"],
            metadata={
                "id":        chunk.get("alarm_id", chunk.get("kpi_id", "N/A")),
                "cell_id":   chunk.get("cell_id", "N/A"),
                "severity":  chunk.get("severity", "N/A"),
                "timestamp": chunk.get("timestamp", ""),
                "doc_type":  doc_type,  # FIXED: Correct doc_type
                "source":    str(json_path),
                "word_count": chunk.get("word_count", 0),
                "probable_cause": chunk.get("probable_cause", ""),  # FIXED: Include probable_cause
                "kpi_name": chunk.get("kpi_name", ""),  # FIXED: Include kpi_name
                "unit": chunk.get("unit", ""),  # FIXED: Include unit
                "value": chunk.get("value", ""),  # FIXED: Include value
            },
        ))

    logger.info(f"parse_oran_data: returning {len(docs)} Documents")
    return docs


def parse_oran_data_streaming(oran_dir: Path = None) -> Generator[Tuple[str, List[Document]], None, None]:
    """
    Generator version for streaming/lazy processing.
    Yields chunks in batches by cell to control memory usage.

    Yields:
        (batch_id, list_of_documents)
    """
    if oran_dir is None:
        oran_dir = RAW_ORAN_DIR
    json_path = oran_dir / "oran_data.json"
    data = load_oran_data(json_path)

    if not data:
        logger.warning("No O-RAN data found — returning empty generator.")
        return

    alarms = data.get('alarms', [])
    kpis = data.get('kpis', [])

    logger.info(f"Streaming {len(alarms)} alarms...")

    # Yield alarms first
    for alarm_idx, alarm in enumerate(tqdm(alarms, desc="Streaming alarms")):
        alarm_id = alarm.get('alarm_id', f"alarm_{alarm_idx}")
        alarm_name = alarm.get('alarm_name', '')
        cell_id = alarm.get('cell_id', 'N/A')
        severity = alarm.get('perceived_severity', 'unknown')
        timestamp = alarm.get('occurrence_time', '')
        probable_cause = alarm.get('probable_cause', '')
        specific_problem = alarm.get('specific_problem', '')
        additional_text = alarm.get('additional_text', '')
        context = alarm.get('context', '')

        content_parts = [
            f"[ALARM] {alarm_id} - {alarm_name}",
            f"Cell ID: {cell_id}",
            f"Severity: {severity}",
            f"Timestamp: {timestamp}",
            f"Probable Cause: {probable_cause}",
            f"Specific Problem: {specific_problem}",
            f"Additional Info: {additional_text}",
            f"Full Context: {context}"
        ]
        content = "\n".join(content_parts)

        doc = Document(
            page_content=content,
            metadata={
                "id": alarm_id,
                "cell_id": cell_id,
                "severity": severity,
                "timestamp": timestamp,
                "doc_type": "oran_alarm",
                "source": str(json_path),
                "word_count": len(content.split()),
                "probable_cause": probable_cause,
                "alarm_name": alarm_name,
            }
        )
        yield f"alarm_{alarm_idx}", [doc]

    logger.info(f"Streaming {len(kpis)} KPIs...")

    # Yield KPIs in batches by cell
    kpis_by_cell = {}
    for kpi in kpis:
        cell_id = kpi.get('cell_id', 'unknown')
        if cell_id not in kpis_by_cell:
            kpis_by_cell[cell_id] = []
        kpis_by_cell[cell_id].append(kpi)

    for cell_id, cell_kpis in tqdm(kpis_by_cell.items(), desc="Streaming KPI batches by cell"):
        docs = []
        for kpi_idx, kpi in enumerate(cell_kpis):
            kpi_name = kpi.get('kpi_name', 'Unknown')
            timestamp = kpi.get('timestamp', 'N/A')
            value = kpi.get('value', 'N/A')
            unit = kpi.get('unit', '')
            context = kpi.get('context', '')

            content_parts = [
                f"[KPI Report] {kpi_name}",
                f"Cell: {cell_id}",
                f"Period: {timestamp}",
                f"Value: {value} {unit}",
                f"Context: {context}"
            ]
            content = "\n".join(content_parts)

            doc = Document(
                page_content=content,
                metadata={
                    "id": kpi.get('id', f'kpi_{kpi_idx}'),
                    "kpi_name": kpi_name,
                    "cell_id": cell_id,
                    "value": value,
                    "unit": unit,
                    "timestamp": timestamp,
                    "doc_type": "oran_kpi",
                    "source": str(json_path),
                    "word_count": len(content.split()),
                }
            )
            docs.append(doc)

        yield f"kpis_{cell_id}", docs
        gc.collect()


def save_oran_chunks(chunks: List[Dict[str, Any]], output_path: Path):
    """Save O-RAN chunks to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in tqdm(chunks, desc="Saving chunks"):
            f.write(json.dumps(chunk) + '\n')

    logger.info(f"✅ Saved {len(chunks)} O-RAN chunks to {output_path}")


def main():
    logger.info("="*70)
    logger.info("O-RAN DATA PROCESSING (FIXED VERSION)")
    logger.info("="*70)

    # Load O-RAN data
    data = load_oran_data(ORAN_JSON)

    if not data:
        logger.warning("No O-RAN data found. Creating demo data...")
        data = {
            'alarms': [
                {
                    'alarm_id': 'ALM-001',
                    'alarm_name': 'RRC_CONNECTION_FAILURE',
                    'perceived_severity': 'HIGH',
                    'probable_cause': 'RRC procedure issue',
                    'specific_problem': 'RRC Connection Failure Rate exceeds threshold',
                    'additional_text': 'Investigate radio link quality',
                    'occurrence_time': '2025-05-23T12:00:00Z',
                    'cell_id': 'CELL_001',
                    'context': 'RRC Connection Failure on CELL_001'
                }
            ],
            'kpis': []
        }

    # Create chunks
    chunks = create_oran_chunks(data)

    # Save
    output_path = PROCESSED_DIR / "oran_chunks.jsonl"
    save_oran_chunks(chunks, output_path)

    logger.info("\n" + "="*70)
    logger.info(f"✅ O-RAN processing complete ({len(chunks)} chunks)")
    logger.info(f"   - Alarms: {len(data.get('alarms', []))}")
    logger.info(f"   - KPIs: {len(data.get('kpis', []))} (ALL processed, no truncation)")
    logger.info("="*70)


if __name__ == "__main__":
    main()
