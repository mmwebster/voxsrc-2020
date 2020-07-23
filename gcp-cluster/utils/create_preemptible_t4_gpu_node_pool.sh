#!/bin/bash

# @param $1 Name of new node pool
if [ -n "$1" ]; then
  SA=${KF_NAME}-vm@${PROJECT}.iam.gserviceaccount.com
  gcloud container node-pools create $1 \
    --cluster=kf-train-6 \
    --enable-autoscaling --max-nodes=2 --min-nodes=0 \
    --disk-size=400GB \
    --num-nodes=0 \
    --preemptible \
    --machine-type=n1-standard-8 \
    --node-taints=preemptible=true:NoSchedule \
    --service-account=$SA \
    --accelerator=type=nvidia-tesla-t4,count=1
else
  echo "Error: Didn't provide node pool name"
fi
