from pathlib import Path
from dataclasses import dataclass


@dataclass
class DrugLabelsConfig:
    api_url: str = "https://api.fda.gov/drug/label.json"
    rxnav_url: str = "https://rxnav.nlm.nih.gov/REST"
    products_path: Path = Path("drugs_to_search_subset.csv")
    candidates_path: Path = Path("rxcui_candidates.csv")
    raw_labels_dir: Path = Path("raw_labels")
    page_size: int = 100
    request_interval_s: float = 0.3  # ~200 requests/min, under openFDA's 240/min limit
