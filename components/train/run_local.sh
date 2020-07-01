#!/bin/bash
# @TODO run this in the components container the same was as defined in the component.yaml
# @param --skip-data-fetch to read local data that was already fetched from GCS
python3 src/voxceleb_trainer/trainSpeakerNet.py --data-bucket=voxsrc-2020-voxceleb-v4 --test_list=vox1_no_cuda.txt --train_list=vox2_no_cuda.txt --test_path=vox1_no_cuda.tar.gz --train_path=vox2_no_cuda.tar.gz --batch_size=5 --max_epoch=1

