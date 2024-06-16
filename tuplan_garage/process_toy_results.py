import argparse
import json
import glob
import os

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str)
args = parser.parse_args()

# Load task successes
with open(args.result_path, 'r') as file:
    samples = json.load(file)

print('=' * 40)
print(f'Saved final results to {args.result_path}')

lane_error = np.mean([sample['lane_error'] for sample in samples])
speed_error = np.mean([sample['speed_error'] for sample in samples])
rewards = np.mean([sample['rewards'] for sample in samples])

print(f'Num episodes: {len(samples)}')
print(f'Average lane error: {lane_error}')
print(f'Average speed error: {speed_error}')
print(f'Average rewards: {rewards}')
print('Done')