# resume-shortlister: AI resume screening in a container.
#
#   docker build -t resume-shortlister .
#   docker run --rm -it -e OPENAI_API_KEY \
#       -v ~/.resume-shortlister:/data -v "$PWD":/work \
#       resume-shortlister chat --resumes ./applicants
#
# The browser app: publish the port on this computer only, then open http://127.0.0.1:8765/
#   docker run --rm -e OPENAI_API_KEY -p 127.0.0.1:8765:8765 \
#       -v ~/.resume-shortlister:/data -v "$PWD":/work resume-shortlister web
#
# /data keeps indexes, uploads and sessions between runs; /work is the folder
# with your resumes and job descriptions, where reports are written. For another
# chat provider, add its extra and pass its settings:
#   docker build --build-arg EXTRAS=anthropic -t resume-shortlister .
#   docker run ... -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY -e EMBEDDER=hashing ...
FROM python:3.13-slim

LABEL org.opencontainers.image.title="Resume Shortlister" \
      org.opencontainers.image.description="Explainable resume-screening assistant" \
      org.opencontainers.image.source="https://github.com/Akshay-Jhijhautia/resume-shortlister" \
      org.opencontainers.image.licenses="MIT"

# Comma-separated pip extras, e.g. anthropic,google; none by default.
ARG EXTRAS=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RESUME_SHORTLISTER_HOME=/data \
    MODELS_DIR=/opt/resume-shortlister/models \
    RESUME_SHORTLISTER_WEB_HOST=0.0.0.0

WORKDIR /opt/resume-shortlister
COPY pyproject.toml README.md ./
COPY resume_shortlister ./resume_shortlister
# Download the local embedding model now, so the container works offline.
RUN pip install ".${EXTRAS:+[$EXTRAS]}" \
    && python -c "from resume_shortlister import config, pools; pools.make_embedder(config.get_settings()).embed(['warm up'])"

RUN useradd --create-home --uid 1000 recruiter \
    && mkdir -p /data /work && chown recruiter /data /work
USER recruiter
WORKDIR /work
VOLUME ["/data"]
EXPOSE 8765

ENTRYPOINT ["resume-shortlister"]
CMD ["--help"]
