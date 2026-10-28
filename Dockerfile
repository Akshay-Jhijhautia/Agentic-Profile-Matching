# profile-match: AI resume screening in a container.
#
#   docker build -t profile-match .
#   docker run --rm -it -e OPENAI_API_KEY \
#       -v ~/.profile-match:/data -v "$PWD":/work \
#       profile-match chat --resumes ./applicants
#
# The browser app: publish the port on this computer only, then open http://127.0.0.1:8765/
#   docker run --rm -e OPENAI_API_KEY -p 127.0.0.1:8765:8765 \
#       -v ~/.profile-match:/data -v "$PWD":/work profile-match web
#
# /data keeps indexes, uploads and sessions between runs; /work is the folder
# with your resumes and job descriptions, where reports are written. For another
# chat provider, add its extra and pass its settings:
#   docker build --build-arg EXTRAS=anthropic -t profile-match .
#   docker run ... -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY -e EMBEDDER=hashing ...
FROM python:3.13-slim

LABEL org.opencontainers.image.title="Profile Match" \
      org.opencontainers.image.description="Explainable resume-screening assistant" \
      org.opencontainers.image.source="https://github.com/Akshay-Jhijhautia/Agentic-Profile-Matching" \
      org.opencontainers.image.licenses="MIT"

# Comma-separated pip extras, e.g. anthropic,google; none by default.
ARG EXTRAS=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PROFILE_MATCH_HOME=/data \
    MODELS_DIR=/opt/profile-match/models \
    PROFILE_MATCH_WEB_HOST=0.0.0.0

WORKDIR /opt/profile-match
COPY pyproject.toml README.md ./
COPY profile_matching ./profile_matching
# Download the local embedding model now, so the container works offline.
RUN pip install ".${EXTRAS:+[$EXTRAS]}" \
    && python -c "from profile_matching import config, pools; pools.make_embedder(config.get_settings()).embed(['warm up'])"

RUN useradd --create-home --uid 1000 recruiter \
    && mkdir -p /data /work && chown recruiter /data /work
USER recruiter
WORKDIR /work
VOLUME ["/data"]
EXPOSE 8765

ENTRYPOINT ["profile-match"]
CMD ["--help"]
