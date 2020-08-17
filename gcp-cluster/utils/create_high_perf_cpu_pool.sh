#!/bin/bash

# @param $1 Name of new node pool
if [ -n "$1" ]; then
  SA=${KF_NAME}-vm@${PROJECT}.iam.gserviceaccount.com
  echo $SA
  gcloud container node-pools create $1 \
    --cluster=kf-train-6 \
    --enable-autoscaling --max-nodes=2 --min-nodes=0 \
    --disk-size=800GB \
    --num-nodes=0 \
    --machine-type=n1-standard-16 \
    --service-account=$SA
else
  echo "Error: Didn't provide node pool name"
fi
