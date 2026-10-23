"""Choosing embedders: local models, their prefixes, and switching a folder's embeddings."""

import shutil
import sys

import numpy as np
import pytest

from profile_matching import pools
from profile_matching.config import Settings
from profile_matching.pools import ResumePool
from profile_matching.rag import job_matcher, resume_rag
from tests.paths import RESUMES_DIR


class FakeEngine:
    """Stands in for a fastembed model: records what it embeds, returns one-hot-ish vectors."""

    def __init__(self):
        self.seen: list[str] = []

    def embed(self, texts, batch_size=32):
        for text in texts:
            self.seen.append(text)
            yield np.array([len(text), 1.0, 0.0], dtype=np.float32)


@pytest.fixture
def engine(monkeypatch):
    fake, loads = FakeEngine(), []
    monkeypatch.setattr(resume_rag, "load_local_model", lambda model, cache_dir: loads.append((model, cache_dir)) or fake)
    fake.loads = loads
    return fake


def test_local_embedder_adds_each_models_prefixes(engine):
    arctic = resume_rag.LocalEmbedder("snowflake/snowflake-arctic-embed-m", cache_dir="/models")
    assert engine.loads == []  # nothing is loaded until something is embedded
    assert arctic.embed(["Python, Django"]) == [[14.0, 1.0, 0.0]]
    arctic.embed_query("Must have: Python")
    assert engine.seen == ["Python, Django", "Represent this sentence for searching relevant passages: Must have: Python"]
    assert engine.loads == [("snowflake/snowflake-arctic-embed-m", "/models")]
    assert arctic.name == "local:snowflake/snowflake-arctic-embed-m"

    engine.seen.clear()
    granite = resume_rag.LocalEmbedder()
    granite.embed(["resume"])
    granite.embed_query("job")
    assert granite.model == "ibm-granite/granite-embedding-small-english-r2" and engine.seen == ["resume", "job"]


def test_every_local_model_is_calibrated():
    # An uncalibrated model would put match scores (and hire decisions) on another scale.
    for model in resume_rag.LOCAL_MODELS:
        assert f"local:{model}" in job_matcher.SEMANTIC_RANGES, model
    low, high = job_matcher.SEMANTIC_RANGES["local:ibm-granite/granite-embedding-small-english-r2"]
    assert job_matcher.calibrate_semantic(low, "local:ibm-granite/granite-embedding-small-english-r2") == 0.0
    assert job_matcher.calibrate_semantic(high, "local:ibm-granite/granite-embedding-small-english-r2") == 1.0
    assert job_matcher.calibrate_semantic(0.5, "openai:text-embedding-3-large") == job_matcher.calibrate_semantic(
        0.5, "openai:text-embedding-3-small")  # by kind when the model has no entry of its own


def test_only_known_local_models_are_accepted(monkeypatch):
    with pytest.raises(ValueError, match="Unsupported local embedding model 'my/model'"):
        resume_rag.LocalEmbedder("my/model")
    with pytest.raises(ValueError, match="Unsupported"):  # measured, but not good enough to offer
        resume_rag.LocalEmbedder("BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "snowflake/snowflake-arctic-embed-m")
    assert resume_rag.LocalEmbedder().model == "snowflake/snowflake-arctic-embed-m"
    assert resume_rag.embedder_from_name("local:snowflake/snowflake-arctic-embed-m", cache_dir="/m").cache_dir == "/m"


def test_make_embedder_follows_the_setting(engine):
    settings = Settings(models_dir="/models")
    assert settings.embedder == "local"
    local = pools.make_embedder(settings)
    assert isinstance(local, resume_rag.LocalEmbedder) and local.cache_dir == "/models"
    assert isinstance(pools.make_embedder(settings, "hashing"), resume_rag.HashingEmbedder)
    assert isinstance(pools.make_embedder(settings, "openai"), resume_rag.OpenAIEmbedder)  # no key needed to build it
    with pytest.raises(ValueError, match="embedder must be local, openai or hashing"):
        Settings(embedder="word2vec")


def test_a_folder_indexed_with_other_embeddings_is_rebuilt(tmp_path):
    folder = tmp_path / "cvs"
    folder.mkdir()
    shutil.copy(RESUMES_DIR / "resume_john_doe.pdf", folder)
    settings = Settings(db_dir=str(tmp_path / "db"), embedder="hashing")
    first = ResumePool.open(folder, settings).sync()
    assert first.reindexed_from is None and "Re-indexed" not in first.summary()

    switched = ResumePool.open(folder, settings, embedder=resume_rag.HashingEmbedder(dimensions=64))
    report = switched.sync()
    assert report.reindexed_from == "hashing:512" and report.ready == 1 and len(report.indexed) == 1
    assert "Re-indexed every resume: the index used other embeddings (hashing:512)." in report.summary()
    assert switched.rag.embedder_name == "hashing:64"
    again = ResumePool.open(folder, settings, embedder=resume_rag.HashingEmbedder(dimensions=64)).sync()
    assert again.reindexed_from is None and again.unchanged == 1


# --- loading the model ------------------------------------------------------------------------------

REAL_LOAD = resume_rag.load_local_model  # the suite replaces it per test (see conftest)


class FakeFastembed:
    """A fastembed module whose model is cached or not, and whose download works or fails."""

    def __init__(self, cached: bool, download_error: Exception | None = None):
        self.cached, self.download_error, self.calls = cached, download_error, []
        fake = self

        class TextEmbedding:
            def __init__(self, model, cache_dir=None, local_files_only=False):
                fake.calls.append((model, cache_dir, local_files_only))
                if local_files_only and not fake.cached:
                    raise ValueError("not in cache")
                if not local_files_only and fake.download_error:
                    raise fake.download_error

        self.TextEmbedding = TextEmbedding


def test_a_cached_model_loads_offline_without_a_download_notice(monkeypatch, tmp_path, capsys):
    fake = FakeFastembed(cached=True)
    monkeypatch.setitem(sys.modules, "fastembed", fake)
    REAL_LOAD("ibm-granite/granite-embedding-small-english-r2", str(tmp_path / "models"))
    assert fake.calls == [("ibm-granite/granite-embedding-small-english-r2", str(tmp_path / "models"), True)]
    assert capsys.readouterr().err == "" and (tmp_path / "models").is_dir()


def test_the_first_load_downloads_and_says_so(monkeypatch, tmp_path, capsys):
    fake = FakeFastembed(cached=False)
    monkeypatch.setitem(sys.modules, "fastembed", fake)
    REAL_LOAD("ibm-granite/granite-embedding-small-english-r2", str(tmp_path))
    assert [call[2] for call in fake.calls] == [True, False]
    assert "Downloading the embedding model ibm-granite/granite-embedding-small-english-r2 (180 MB, once)" in capsys.readouterr().err

    offline = FakeFastembed(cached=False, download_error=ConnectionError("no network"))
    monkeypatch.setitem(sys.modules, "fastembed", offline)
    with pytest.raises(ValueError, match="Could not download the embedding model .*no network.*EMBEDDER=openai"):
        REAL_LOAD("ibm-granite/granite-embedding-small-english-r2", str(tmp_path))
