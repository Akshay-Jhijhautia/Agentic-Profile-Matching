import pytest

from profile_matching.candidates import CandidateDirectory, CandidateNotFoundError, candidate_id


class StubRAG:
    def __init__(self, names):
        self.calls = 0
        self.records = [
            {"candidate_name": name, "resume_path": f"/r/resume_{name.lower().replace(' ', '_')}.pdf",
             "experience_years": 3.0, "highest_degree": None, "skills": ["Python"]}
            for name in names
        ]

    def list_resumes(self):
        self.calls += 1
        return self.records


@pytest.fixture
def directory():
    return CandidateDirectory(StubRAG(["Maria Garcia", "Mario Rossi", "Jane Smith", "Jane Doe"]))


def test_candidate_id_is_file_stem_without_prefix():
    assert candidate_id("/x/resumes/resume_maria_garcia.pdf") == "maria_garcia"
    assert candidate_id("cv_bob.docx") == "cv_bob"
    assert candidate_id("/x/Jordan_Rivera_CV_scan.pdf") == "jordan_rivera_cv_scan"


@pytest.mark.parametrize(
    "ref",
    ["maria_garcia", "Maria Garcia", "  maria-garcia ", "MARIA", "garcia",
     "resume_maria_garcia", "resumes/resume_maria_garcia.pdf"],
)
def test_resolve_accepts_ids_names_and_paths(directory, ref):
    assert directory.resolve(ref).name == "Maria Garcia"


def test_resolve_reports_ambiguity_with_ids(directory):
    with pytest.raises(CandidateNotFoundError, match="jane_smith.*jane_doe|jane_doe.*jane_smith"):
        directory.resolve("Jane")


def test_resolve_suggests_close_names(directory):
    with pytest.raises(CandidateNotFoundError, match="Did you mean: maria garcia"):
        directory.resolve("Marla Garcia")


@pytest.mark.parametrize("ref", ["", "   "])
def test_resolve_rejects_empty(directory, ref):
    with pytest.raises(CandidateNotFoundError):
        directory.resolve(ref)


def test_resolve_many_drops_repeats_and_keeps_order(directory):
    profiles = directory.resolve_many(["Jane Doe", "maria", "maria_garcia"])
    assert [p.id for p in profiles] == ["jane_doe", "maria_garcia"]


def test_profiles_are_loaded_once_until_refresh():
    rag = StubRAG(["Maria Garcia"])
    directory = CandidateDirectory(rag)
    directory.resolve("maria")
    directory.resolve("garcia")
    assert rag.calls == 1
    directory.refresh()
    directory.resolve("maria")
    assert rag.calls == 2


def test_resume_text_reads_the_file(services):
    profile = services.candidates.resolve("Alex Lee")
    assert "Alex Lee" in services.candidates.resume_text(profile)
    assert profile.summary()["resume"].endswith("resume_alex_lee.txt")
