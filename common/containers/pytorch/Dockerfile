# Image for training pytorch models w/ GPU(s)

# comes with pytorch 1.4.0 and cuda 10.1
FROM gcr.io/deeplearning-platform-release/pytorch-gpu

# install voxceleb_trainer dependencies
RUN apt-get -y update && apt-get install -y libzbar-dev
RUN pip install --upgrade pip
# downgrade conda--hacky workaround for "failed on initial frozen solve"
RUN conda install conda=4.6

# install pytorch with explicit version of cuda 10.1 (default is 10.2, which is
# incompatible with GKE nvidia drivers installed with google's daemon installer)
RUN conda install pytorch=1.6 cudatoolkit=10.1 -c pytorch

RUN pip install --no-cache-dir torchaudio numpy \
    scipy scikit-learn tqdm PyYAML wandb

# python GCP client
RUN pip install --upgrade --no-cache-dir google-cloud-storage

# install ffmpeg audio compression tool
RUN conda install -c conda-forge ffmpeg
RUN conda install -c conda-forge parallel
RUN conda install -c anaconda pigz
