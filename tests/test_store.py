"""The SQLite + NumPy vector store on its own."""

import sqlite3
import threading

import numpy as np
import pytest

from resume_shortlister.rag.vector_store import DB_FILE, Chunk, VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "index")


def _chunks(prefix: str, count: int, resume: str = "a.pdf") -> list[Chunk]:
    return [Chunk(f"{prefix}{i}", f"text {prefix}{i}", {"resume_path": resume, "n": i}) for i in range(count)]


def test_query_returns_the_exact_nearest_chunks(store):
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(300, 64)).astype(np.float32)
    resumes = [f"r{i % 30}.pdf" for i in range(300)]
    collection = store.create_collection("pool", {"embedder": "test"})
    collection.add([Chunk(f"c{i}", f"doc {i}", {"resume_path": resumes[i]}) for i in range(300)], vectors.tolist())

    query = rng.normal(size=64)
    expected = vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
    hits = collection.query(query.tolist(), n_results=20)
    assert [h.chunk.id for h in hits] == [f"c{i}" for i in np.argsort(-expected, kind="stable")[:20]]
    assert [h.similarity for h in hits] == pytest.approx(sorted(expected, reverse=True)[:20], abs=1e-5)

    # Restricted to some resumes: every one of their chunks, and nothing else.
    hits = collection.query(query.tolist(), n_results=1000, resume_paths=["r1.pdf", "r2.pdf"])
    assert len(hits) == 20 and {h.chunk.metadata["resume_path"] for h in hits} == {"r1.pdf", "r2.pdf"}
    assert collection.query(query.tolist(), n_results=5, resume_paths=["missing.pdf"]) == []


def test_equal_similarities_keep_insertion_order(store):
    collection = store.create_collection("pool", {})
    collection.add(_chunks("c", 5), [[1.0, 0.0]] * 5)
    assert [h.chunk.id for h in collection.query([2.0, 0.0], n_results=5)] == ["c0", "c1", "c2", "c3", "c4"]


def test_edge_cases_are_safe(store):
    collection = store.create_collection("pool", {})
    assert collection.query([1.0, 0.0], n_results=3) == [] and collection.count() == 0
    collection.add(_chunks("c", 2), [[1.0, 0.0], [0.0, 0.0]])  # a zero vector stays at similarity 0
    assert [round(h.similarity, 6) for h in collection.query([0.0, 3.0], n_results=5)] == [0.0, 0.0]
    assert collection.query([0.0, 0.0], n_results=5)[0].similarity == 0.0
    with pytest.raises(ValueError, match="3 dimensions; this collection has 2"):
        collection.add(_chunks("d", 1), [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="Query has 3 dimensions"):
        collection.query([1.0, 2.0, 3.0], n_results=1)
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        collection.add(_chunks("e", 2), [[1.0, 0.0]])
    with pytest.raises(sqlite3.IntegrityError):  # ids are unique within a collection
        collection.add(_chunks("c", 1), [[1.0, 0.0]])
    assert collection.count() == 2  # a failed add changes nothing


def test_records_update_delete_and_order(store):
    collection = store.create_collection("pool", {"folder": "/x"})
    collection.add(_chunks("a", 2, "a.pdf") + _chunks("b", 2, "b.pdf"), [[1.0, 0.0]] * 4)
    assert [c.id for c in collection.chunks()] == ["a0", "a1", "b0", "b1"]
    assert [c.id for c in collection.chunks("b.pdf", limit=1)] == ["b0"]

    collection.update_metadata("a.pdf", {"file_stamp": "s2"})
    assert [c.metadata for c in collection.chunks("a.pdf")] == [
        {"resume_path": "a.pdf", "n": 0, "file_stamp": "s2"}, {"resume_path": "a.pdf", "n": 1, "file_stamp": "s2"}]
    collection.delete("a.pdf")
    assert [c.id for c in collection.chunks()] == ["b0", "b1"]
    assert [h.chunk.id for h in collection.query([1.0, 0.0], n_results=10)] == ["b0", "b1"]


def test_collections_persist_and_stay_separate(tmp_path):
    first = VectorStore(tmp_path / "index")
    one = first.create_collection("pool-one", {"folder": "/one"})
    two = first.create_collection("pool-two", {"folder": "/two"})
    one.add(_chunks("c", 3), [[1.0, 0.0]] * 3)
    two.add(_chunks("c", 1), [[0.0, 1.0]])  # the same id in another collection is fine
    first.close()

    reopened = VectorStore(tmp_path / "index")
    assert (tmp_path / "index" / DB_FILE).exists()
    assert [(c.name, c.metadata) for c in reopened.list_collections()] == [
        ("pool-one", {"folder": "/one"}), ("pool-two", {"folder": "/two"})]
    assert reopened.get_collection("pool-one").count() == 3 and reopened.get_collection("nope") is None
    reopened.delete_collection("pool-one")
    assert [c.name for c in reopened.list_collections()] == ["pool-two"]
    assert reopened.get_collection("pool-two").count() == 1


def test_searches_see_changes_made_elsewhere(tmp_path):
    # Another process (here: another connection) indexes while this one keeps its matrix cached.
    reader = VectorStore(tmp_path / "index").create_collection("pool", {})
    reader.add(_chunks("a", 1), [[1.0, 0.0]])
    assert len(reader.query([1.0, 0.0], n_results=10)) == 1
    VectorStore(tmp_path / "index").get_collection("pool").add(_chunks("b", 1), [[1.0, 0.0]])
    assert len(reader.query([1.0, 0.0], n_results=10)) == 2

    # And another handle on the same collection through the same store.
    store = VectorStore(tmp_path / "other")
    first = store.create_collection("pool", {})
    first.add(_chunks("a", 1), [[1.0, 0.0]])
    assert len(first.query([1.0, 0.0], n_results=10)) == 1
    store.get_collection("pool").add(_chunks("b", 1), [[1.0, 0.0]])
    assert len(first.query([1.0, 0.0], n_results=10)) == 2


def test_parallel_searches_and_writes(store):
    collection = store.create_collection("pool", {})
    collection.add(_chunks("seed", 10), [[1.0, 0.0]] * 10)
    errors: list[Exception] = []

    def search():
        try:
            for _ in range(50):
                assert len(collection.query([1.0, 0.0], n_results=5)) == 5
        except Exception as exc:  # noqa: BLE001  (reported below)
            errors.append(exc)

    def write():
        for i in range(20):
            collection.add([Chunk(f"w{i}", "t", {"resume_path": f"w{i}.pdf"})], [[0.0, 1.0]])

    threads = [threading.Thread(target=search) for _ in range(6)] + [threading.Thread(target=write)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and collection.count() == 30


def test_an_index_from_a_newer_version_is_refused(tmp_path):
    VectorStore(tmp_path / "index").close()
    with sqlite3.connect(tmp_path / "index" / DB_FILE) as db:
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(ValueError, match="newer resume-shortlister"):
        VectorStore(tmp_path / "index")


def test_a_new_database_waits_for_another_process_instead_of_failing(tmp_path):
    from resume_shortlister.rag.vector_store import open_sqlite

    path = tmp_path / "new.sqlite"
    other = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    other.execute("CREATE TABLE t(v)")
    other.execute("BEGIN")
    other.execute("SELECT * FROM t").fetchall()  # holds the database while we switch it to WAL
    threading.Timer(0.5, lambda: other.execute("COMMIT")).start()
    db = open_sqlite(path)
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_several_processes_can_create_the_same_stores_at_once(tmp_path):
    """Processes switching the same new database to WAL at the same instant once failed with
    "database is locked" (the web app and a terminal screening started together)."""
    import subprocess
    import sys
    import time

    script = (
        "import pathlib, sys\n"
        "from resume_shortlister.llm_cache import ResponseCache\n"
        "from resume_shortlister.rag.vector_store import VectorStore\n"
        "while not pathlib.Path(sys.argv[1]).exists(): pass\n"
        "ResponseCache(sys.argv[2]); VectorStore(sys.argv[3])\n"
    )
    for round_ in range(2):
        start, folder = tmp_path / f"go{round_}", tmp_path / f"home{round_}"
        procs = [subprocess.Popen([sys.executable, "-c", script, str(start), str(folder / "answers.sqlite"),
                                   str(folder / "index")], stderr=subprocess.PIPE) for _ in range(12)]
        time.sleep(1)  # every process waiting on the start signal
        start.touch()
        errors = [p.communicate(timeout=120)[1].decode() for p in procs]
        assert [p.returncode for p in procs] == [0] * 12, [e for e in errors if e][:1]
