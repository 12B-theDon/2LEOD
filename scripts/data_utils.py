import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_scan_dataset(feature_json: str) -> List[Dict]:
    path = Path(feature_json)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("frames", [])


def split_frames(dataset: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    ordered = dataset.copy()
    rng.shuffle(ordered)
    split_index = int(len(ordered) * train_ratio)
    return ordered[:split_index], ordered[split_index:]
