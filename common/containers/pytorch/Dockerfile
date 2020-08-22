# Image for training pytorch models w/ GPU(s)
# Copy from another project so it's available in personal container registry

FROM anibali/pytorch:1.5.0-cuda10.2

# install voxceleb_trainer dependencies
RUN sudo apt-get -y update && sudo apt-get install -y libzbar-dev
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch~=1.6.0 torchaudio~=0.5.0 numpy \
    scipy scikit-learn tqdm PyYAML wandb

# python GCP client
RUN pip install --upgrade --no-cache-dir google-cloud-storage

# install ffmpeg audio compression tool
RUN conda install -c conda-forge ffmpeg
RUN conda install -c conda-forge parallel
RUN conda install -c anaconda pigz
