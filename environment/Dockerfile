FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    libosmesa6-dev libglfw3 patchelf \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install dependencies first (better layer caching on rebuilds)
COPY environment/requirements.txt ./environment/requirements.txt
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    pip install --no-cache-dir -r environment/requirements.txt && \
    pip install --no-cache-dir "numpy==1.26.4" --force-reinstall

# Repository layout (see README.md for what each folder contains)
COPY core/ ./core/
COPY configs/ ./configs/
COPY training/ ./training/
COPY deployment/ ./deployment/
COPY test_sequence/ ./test_sequence/
COPY analysis/ ./analysis/
COPY patches/ ./patches/
COPY legacy/ ./legacy/
COPY figures/ ./figures/
COPY experiment_artifacts/ ./experiment_artifacts/

ENV MUJOCO_GL=osmesa
ENV PYTHONPATH=/app

# No single "run everything" entry point — this framework is invoked per
# config, e.g.:
#   docker run --rm -v $(pwd)/results:/app/results audit \
#       python core/launch_audit.py configs/hopper_vc.json
CMD ["python3", "core/launch_audit.py", "--help"]
