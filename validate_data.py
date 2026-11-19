"""
SupportIQ - Dataset Validation Module
Verifies dataset integrity, label validity, split distributions,
and ensures zero data leakage across train, validation, and test splits.

Author: SupportIQ ML Engineering Team
"""

import json
import os
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple


ALLOWED_CATEGORIES = {"billing", "technical", "account", "shipping"}
ALLOWED_PRIORITIES = {"low", "medium", "high"}
ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}


def load_and_validate_jsonl(filepath: str) -> Tuple[List[Dict], Counter, Counter, Counter]:
    """
    Loads and validates a single JSONL split file.
    Returns records and distribution counters.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required dataset file: {filepath}")

    records = []
    cat_dist = Counter()
    prio_dist = Counter()
    sent_dist = Counter()
    seen_ids = set()

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num} in {filepath} is invalid JSON: {e}")

            # Check required fields
            for field in ["id", "instruction", "input", "output"]:
                if field not in record:
                    raise KeyError(f"Line {line_num} missing required field '{field}'")
            
            if not isinstance(record["input"], str) or len(record["input"].strip()) == 0:
                raise ValueError(f"Line {line_num} has empty or non-string input.")

            # Validate ID uniqueness within split
            rec_id = record["id"]
            if rec_id in seen_ids:
                raise ValueError(f"Duplicate ID found within {filepath}: {rec_id}")
            seen_ids.add(rec_id)

            # Validate output field is valid JSON matching allowed labels
            try:
                parsed_output = json.loads(record["output"])
            except json.JSONDecodeError:
                raise ValueError(f"Line {line_num} output is not valid JSON string: {record['output']}")

            cat = parsed_output.get("category")
            prio = parsed_output.get("priority")
            sent = parsed_output.get("sentiment")

            if cat not in ALLOWED_CATEGORIES:
                raise ValueError(f"Invalid category '{cat}' on line {line_num}. Allowed: {ALLOWED_CATEGORIES}")
            if prio not in ALLOWED_PRIORITIES:
                raise ValueError(f"Invalid priority '{prio}' on line {line_num}. Allowed: {ALLOWED_PRIORITIES}")
            if sent not in ALLOWED_SENTIMENTS:
                raise ValueError(f"Invalid sentiment '{sent}' on line {line_num}. Allowed: {ALLOWED_SENTIMENTS}")

            cat_dist[cat] += 1
            prio_dist[prio] += 1
            sent_dist[sent] += 1
            records.append(record)

    return records, cat_dist, prio_dist, sent_dist


def check_data_leakage(train_recs: List[Dict], val_recs: List[Dict], test_recs: List[Dict]):
    """
    Ensures zero text overlap between train, validation, and test splits.
    """
    train_texts = {r["input"].strip().lower() for r in train_recs}
    val_texts = {r["input"].strip().lower() for r in val_recs}
    test_texts = {r["input"].strip().lower() for r in test_recs}

    leak_train_val = train_texts.intersection(val_texts)
    leak_train_test = train_texts.intersection(test_texts)
    leak_val_test = val_texts.intersection(test_texts)

    if leak_train_val:
        raise ValueError(f"DATA LEAKAGE DETECTED! {len(leak_train_val)} overlapping samples between train and validation.")
    if leak_train_test:
        raise ValueError(f"DATA LEAKAGE DETECTED! {len(leak_train_test)} overlapping samples between train and test.")
    if leak_val_test:
        raise ValueError(f"DATA LEAKAGE DETECTED! {len(leak_val_test)} overlapping samples between validation and test.")

    print("Data leakage check PASSED: 0 overlapping examples across splits.")


def validate_all(data_dir: str):
    print("=" * 60)
    print("SupportIQ - Dataset Validation Suite")
    print("=" * 60)
    print(f"Data Directory: {data_dir}\n")

    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "validation.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")

    train_recs, tr_cat, tr_prio, tr_sent = load_and_validate_jsonl(train_path)
    val_recs, v_cat, v_prio, v_sent = load_and_validate_jsonl(val_path)
    test_recs, te_cat, te_prio, te_sent = load_and_validate_jsonl(test_path)

    total_samples = len(train_recs) + len(val_recs) + len(test_recs)
    print(f"Dataset Size Summary:")
    print(f"  - Train samples:      {len(train_recs):4d} ({len(train_recs)/total_samples*100:.1f}%)")
    print(f"  - Validation samples: {len(val_recs):4d} ({len(val_recs)/total_samples*100:.1f}%)")
    print(f"  - Test samples:       {len(test_recs):4d} ({len(test_recs)/total_samples*100:.1f}%)")
    print(f"  - Total samples:      {total_samples:4d}\n")

    # Minimum constraints
    if total_samples < 1000:
        raise ValueError(f"Total samples ({total_samples}) is below the required 1,000 threshold!")
    if len(test_recs) < 50:
        raise ValueError(f"Held-out test set size ({len(test_recs)}) is below the required 50 threshold!")

    print("Distribution Verification (Test Split):")
    print(f"  Categories:  {dict(te_cat)}")
    print(f"  Priorities:  {dict(te_prio)}")
    print(f"  Sentiments:  {dict(te_sent)}\n")

    # Data leakage check
    check_data_leakage(train_recs, val_recs, test_recs)
    print("\nALL DATASET VALIDATION CHECKS PASSED SUCCESSFULLY (Exit Code 0).")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_directory = os.path.join(base_dir, "data")
    try:
        validate_all(data_directory)
    except Exception as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)

