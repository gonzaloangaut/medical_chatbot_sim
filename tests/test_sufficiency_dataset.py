"""Structural regression checks for the evidence-sufficiency annotations."""

import copy
import json
from pathlib import Path

import pytest

from evaluation.validate_sufficiency_dataset import DatasetValidationError, validate

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"


def load_file(path):
    """
    Load a JSON file and return its contents as a Python object.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs():
    """
    Load the four input files needed for the sufficiency dataset validation 
    tests.
    """
    return (
        load_file(EVALUATION / "datasets" / "sufficiency_dev.json"),
        load_file(EVALUATION / "datasets" / "chunk_catalog.json"),
        load_file(EVALUATION / "datasets" / "retrieval_dev.json"),
        load_file(EVALUATION / "results" / "retrieval" / "retrieval_dev.json"),
    )


def test_development_sufficiency_annotations_are_structurally_valid():
    """
    Test that the sufficiency dataset annotations are structurally valid and 
    consistent with the frozen retrieval results.
    """
    dataset, catalog, dev, rankings = load_inputs()
    counts = validate(dataset, catalog, dev, rankings)
    assert counts == {"SUFFICIENT": 10, "PARTIAL": 10, "INSUFFICIENT": 10}


def test_rejects_a_support_reference_outside_supplied_evidence():
    """
    Test that the validation function raises an error when a supporting chunk 
    ID is provided that is not included in the evidence chunk IDs for a 
    requirement.
    """
    dataset, catalog, dev, rankings = load_inputs()
    altered = copy.deepcopy(dataset)
    altered["cases"][10]["requirements"][0]["supporting_chunk_ids"] = [
        "medlineplus_fever_000"
    ]
    with pytest.raises(DatasetValidationError, match="points outside provided evidence"):
        validate(altered, catalog, dev, rankings)


def test_rejects_inconsistent_global_sufficiency_label():
    """
    Test that the validation function raises an error when the global sufficiency
    label for a case is inconsistent with the sufficiency labels of its
    requirements.
    """
    dataset, catalog, dev, rankings = load_inputs()
    altered = copy.deepcopy(dataset)
    altered["cases"][10]["expected_sufficiency"] = "SUFFICIENT"
    with pytest.raises(DatasetValidationError, match="label inconsistent"):
        validate(altered, catalog, dev, rankings)


def test_rejects_drift_in_frozen_retrieval_candidates():
    """
    Test that the validation function raises an error when the evidence chunk IDs
    for a case are reordered, indicating a drift from the frozen retrieval results.
    """
    dataset, catalog, dev, rankings = load_inputs()
    altered = copy.deepcopy(dataset)
    altered["cases"][0]["evidence_chunk_ids"].reverse()
    with pytest.raises(DatasetValidationError, match="evidence differs from frozen"):
        validate(altered, catalog, dev, rankings)
