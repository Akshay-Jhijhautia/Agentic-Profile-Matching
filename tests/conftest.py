import pytest

from profile_matching.config import Settings
from profile_matching.rag import job_matcher, resume_rag
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


@pytest.fixture
def services(hashing_db, tmp_path) -> Services:
    """Services over the hashing index, writing reports to a temporary directory.

    Tests that need the LLM set ``services.model`` to a ``FakeChatModel``.
    """
    settings = Settings(db_dir=hashing_db, reports_dir=str(tmp_path / "reports"))
    return Services(rag=resume_rag.ResumeRAG(hashing_db), settings=settings)
