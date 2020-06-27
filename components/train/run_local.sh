#!/bin/bash
# @TODO run this in the components container the same was as defined in the component.yaml
python3 src/voxceleb_trainer/trainSpeakerNet.py --data-bucket=voxsrc-2020-voxceleb-v4 --test_list=vox1_test_list_small.txt --train_list=vox2_train_list_small.txt --test_path=voxceleb1-small-wav.tar.gz --train_path=voxceleb2-small-m4a.tar.gz --skip-data-fetch=True --batch_size=30 --max_epoch=1

