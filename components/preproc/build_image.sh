#!/bin/bash -e
image_name=gcr.io/voxsrc-2020-dev-1/preprocessor
image_tag=latest
full_image_name=${image_name}:${image_tag}

# copy project src to local temp dir
mkdir -p ./build/
cp -r ./src ./build/src

# move into container's dir
cd "$(dirname "$0")"
# build it
docker build -t "${full_image_name}" .
# upload it
docker push "$full_image_name"

# Output the strict image name (which contains the sha256 image digest)
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"

# delete temp copy of src
rm -rf ./build
