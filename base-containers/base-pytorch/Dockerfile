# Image for training pytorch models w/ GPU(s)
# Copy from another project so it's available in personal container registry

FROM anibali/pytorch:1.5.0-cuda9.2

# install voxceleb_trainer dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==1.5.0 torchaudio==0.5.0 numpy \
    scipy scikit-learn tqdm

# python GCP client
RUN pip install --upgrade --no-cache-dir google-cloud-storage

# install ffmpeg audio compression tool
RUN conda install -c conda-forge ffmpeg
RUN conda install -c conda-forge parallel
