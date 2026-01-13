# =============================================================================
# AbXtract Dockerfile
# Antibody Descriptor Calculator - Production Docker Image
# =============================================================================

FROM continuumio/miniconda3:23.10.0-1

LABEL maintainer="AbXtract Team"
LABEL description="AbXtract - Comprehensive antibody descriptor calculator"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create conda environment
COPY abxtract.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml -n abxtract \
    && conda clean -afy \
    && rm /tmp/environment.yml

# Activate conda environment by default
SHELL ["conda", "run", "-n", "abxtract", "/bin/bash", "-c"]


# Install reduce from bioconda (if not already in env)
# RUN conda install -y -n abxtract -c bioconda reduce || true
# RUN conda activate abxtract

RUN pip install abnumber propka
RUN conda install bioconda::hmmer || y
RUN conda install -c salilab dssp -y # i had difficulties making dssp work  || true
RUN pip install peptides protpy prody tqdm numba matplotlib
RUN pip install scikit-learn
RUN pip install seaborn 
RUN conda install bioconda::anarci  || true
RUN pip install freesasa
RUN conda install conda-forge::openmm  || true
RUN conda install conda-forge::pdbfixer  || true
RUN pip install pdbe-arpeggio
RUN conda install bioconda::muscle  || true
RUN conda install bioconda::reduce  || true
RUN pip install ipykernel
RUN python -m ipykernel install --user --name=abxtract --display-name "abxtract"




# Create application directory
WORKDIR /app

# Copy AbXtract package
COPY AbXtract/ /app/AbXtract/
COPY run_abxtract.py /app/run_abxtract.py

# Create directories for data I/O
RUN mkdir -p /data/test /data/output /app/temp

# Set permissions
RUN chmod +x /app/run_abxtract.py

# Create entrypoint script
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /entrypoint.sh && \
    echo 'conda activate abxtract' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh
    
# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command shows help
CMD ["python", "/app/run_abxtract.py", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD conda run -n abxtract python -c "from AbXtract import AntibodyDescriptorCalculator; print('OK')" || exit 1
