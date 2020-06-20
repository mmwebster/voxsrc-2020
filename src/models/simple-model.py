#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import subprocess

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument('--input1-path', type=str)
# kubeflow autogens this output path
parser.add_argument('--model-save-path', type=str)
args = parser.parse_args()

print(f"Received upstream data via path: {args.input1_path}")
print(f"Printing contents...")
with open(args.input1_path, 'r') as input1_file:
    for x, line in enumerate(input1_file):
        print(f"{x}: {line}")

print(f"Cuda availability: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

print(f"Saving trained model to: {args.model_save_path}")
# touch the output file/dir
Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True)
# write contents
with open(args.model_save_path, 'w') as model_save_file:
    model_save_file.write("[model]\n")
