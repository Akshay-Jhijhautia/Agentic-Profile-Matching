"""The retrieval engine the agent builds on.

- ``fs_tools``: read, list, write and search resume files (PDF, DOCX, TXT, MD)
- ``resume_rag``: section chunking, metadata extraction, indexing
- ``vector_store``: chunks, metadata and embeddings in SQLite; exact cosine search with NumPy
- ``job_matcher``: JD parsing, hybrid search, must-have checks, scoring
- ``skills_vocab``: skill aliases, free-text detection, implied skills
"""
