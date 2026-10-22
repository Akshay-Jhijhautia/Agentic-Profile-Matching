# profile-match: AI resume screening in a container.
#
#   docker build -t profile-match .
#   docker run --rm -it -e OPENAI_API_KEY \
#       -v ~/.profile-match:/data -v "$PWD":/work \
#       profile-match chat --resumes ./applicants
#
# /data keeps indexes, uploads and sessions between runs; /work is the folder
# with your resumes and job descriptions, where reports are written.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PROFILE_MATCH_HOME=/data

WORKDIR /opt/profile-match
COPY pyproject.toml README.md ./
COPY profile_matching ./profile_matching
# Install with OCR, and download its models now so the container works offline.
RUN pip install ".[ocr]" \
    && python -c "from profile_matching.rag import ocr; ocr._engine()"

RUN useradd --create-home --uid 1000 recruiter \
    && mkdir -p /data /work && chown recruiter /data /work
USER recruiter
WORKDIR /work
VOLUME ["/data"]

ENTRYPOINT ["profile-match"]
CMD ["--help"]
