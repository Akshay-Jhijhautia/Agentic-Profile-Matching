from collections import Counter

import pytest

from profile_matching.rag import resume_rag
from scripts import generate_sample_resumes as generator
from scripts import synthetic_resumes

FAMILIES = {family.key: family for family in synthetic_resumes.FAMILIES}


@pytest.fixture(scope="module")
def dataset():
    return generator.build_dataset()


@pytest.fixture(scope="module")
def parsed(tmp_path_factory, dataset):
    """Metadata extracted (rules only) from every generated file, keyed by stem."""
    directory = tmp_path_factory.mktemp("resumes")
    generator.generate(directory)
    return {
        r["file_name"].rsplit(".", 1)[0]: resume_rag.extract_metadata(r["content"], r["path"])
        for r in resume_rag.load_resumes(str(directory))
    }


def test_dataset_has_handwritten_resumes_first_then_synthetic(dataset):
    assert len(dataset) == generator.DEFAULT_COUNT
    assert list(dataset)[: len(generator.RESUMES)] == list(generator.RESUMES)
    names = [data["name"] for data in dataset.values()]
    assert len(set(names)) == len(names)


def test_generation_is_deterministic_per_seed(dataset):
    assert generator.build_dataset() == dataset
    assert generator.build_dataset(seed=7) != dataset


def test_families_and_formats_are_balanced(dataset):
    synthetic = [d for d in dataset.values() if "family" in d]
    families = Counter(d["family"] for d in synthetic)
    formats = Counter(d["format"] for d in synthetic)
    assert set(families) == set(FAMILIES)
    assert max(families.values()) - min(families.values()) <= 1
    assert max(formats.values()) - min(formats.values()) <= 1


def test_count_below_handwritten_is_rejected():
    with pytest.raises(ValueError):
        generator.build_dataset(count=len(generator.RESUMES) - 1)


def test_every_generated_file_is_readable(parsed):
    assert len(parsed) == generator.DEFAULT_COUNT


def test_synthetic_resumes_parse_back(dataset, parsed):
    for stem, data in dataset.items():
        if "family" not in data:
            continue
        meta = parsed[stem]
        assert meta["name"] == data["name"], stem
        assert meta["experience_years"] and meta["experience_years"] > 0, stem
        assert set(FAMILIES[data["family"]].core) <= set(meta["skills"]), stem


def test_dataset_spans_seniority_and_education(parsed):
    years = [meta["experience_years"] for meta in parsed.values()]
    degrees = Counter(meta["highest_degree"] for meta in parsed.values())
    assert min(years) < 3 and max(years) > 10
    assert {"Bachelor's", "Master's", "PhD", None} <= set(degrees)
