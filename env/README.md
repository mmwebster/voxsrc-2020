# Local Development Setup
* [install conda](https://docs.conda.io/en/latest/miniconda.html)
* create the conda environment `conda env create -f  conda-vox-env-[YOUR PLATFORM].yaml`
* activate the conda environment `conda activate voxsrc-2020`
* [setup authentication for the python client](https://cloud.google.com/docs/authentication/getting-started#auth-cloud-implicit-python)

# Updating ENV with new Conda .yaml
* run `conda activate [existing-env]`
* run `conda env update --file [updated-env.yml]`
