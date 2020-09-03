# VoxSRC 2020 Speaker Recognition Challenge
Our codebase for the 2020 VoxSRC Speaker Recognition Challenge. It contains source for our ML pipeline and GKE training infrastructure.

Please contact @mmwebster on LinkedIn if you're interested in collaborating on this project.

## Usage
See the wiki for setup instructions. Once setup, you can either
* run KubeFlow components locally (individually) with bash scripts at their root dirs
* or build components' Docker images and run the full pipeline on GKE

Note that the codebase is currently hardcoded to our own personal GCP resource URIs, but it should be easy to switch over to another GCP account or other cloud platform.

## Project Structure
- **common**: any containers, source, or utilities common to multiple components
- **components**: individual stages of the ML pipeline (preprocessing, training, and (at some point in the future) analysis)
  - **[component]**
    - **src**: All python source code for the component
    - **run_local.sh**: Bash script to run the component in a *relatively* similar way as on the cluster
    - **build_image.sh**: Bash script to build the docker image from the dockerfile and push it to our google container registry
    - **Dockerfile**
    - **[component]_component.yaml**: Kubeflow component config, including entry script and I/O artifacts
- **data**: utilities for local download/install of Voxceleb 1 & 2 (from our GCS bucket)
- **env**: conda environments to support local runs (haven't setup local Docker runs, but hopefully soon)
- **gcp-cluster**: bash scripts containing the config used to setup various GCP node pools in our training cluster
- **[pipeline]_pipeline.yaml**: The files specifying which components, with what configs, with what resources are run

## License
Free, open, etc., with the exception of licenses corresponding to baseline code from [clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer). All code originating from this repo is tracked as such in git, with the exception of some manually copied code that is marked by "@credit" comments at the start of their blocks.
