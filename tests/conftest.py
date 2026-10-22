import pytest

from profile_matching.config import Settings
from profile_matching.rag import job_matcher, resume_rag
from profile_matching.pools import ResumePool
from profile_matching.services import Services
from tests.paths import RESUMES_DIR


@pytest.fixture(scope="session")
def hashing_db(tmp_path_factory) -> str:
    """A ChromaDB index of the fixture resumes built with the offline hashing embedder."""
    path = str(tmp_path_factory.mktemp("hashing_db"))
    resume_rag.ResumeRAG(path, embedder=resume_rag.HashingEmbedder()).index_directory(str(RESUMES_DIR))
    return path


@pytest.fixture(scope="session")
def matcher(hashing_db) -> job_matcher.JobMatcher:
    return job_matcher.JobMatcher(resume_rag.ResumeRAG(hashing_db))


@pytest.fixture(scope="session")
def fixture_pool_db(tmp_path_factory) -> str:
    """A database holding the fixture resumes' folder pool, built once with the hashing embedder."""
    path = str(tmp_path_factory.mktemp("pool_db"))
    ResumePool.open(RESUMES_DIR, Settings(db_dir=path, embedder="hashing")).sync()
    return path


@pytest.fixture
def services(fixture_pool_db, tmp_path) -> Services:
    """Services with the fixture resumes' pool active (offline hashing embedder), writing
    reports to a temporary directory.

    Tests that need the LLM set ``services.model`` to a ``FakeChatModel``.
    """
    settings = Settings(db_dir=fixture_pool_db, reports_dir=str(tmp_path / "reports"), embedder="hashing",
                        uploads_dir=str(tmp_path / "uploads"))
    return Services(pool=ResumePool.open(RESUMES_DIR, settings), settings=settings)
