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

keys = [
    'drivable_area_compliance', 
    'driving_direction_compliance', 
    'ego_is_comfortable', 
    'ego_is_making_progress', 
    'ego_progress_along_expert_route', 
    'no_ego_at_fault_collisions', 
    'speed_limit_compliance', 
    'time_to_collision_within_bound', 
    'score'
]
for sample in samples:
    metric_path = glob.glob(f"{sample['metric_path']}/aggregator_metric/*.parquet")[0]
    df = pd.read_parquet(metric_path)
    sample['metrics'] = {key: df[key][0] for key in keys}

with open(args.result_path, 'w') as file:
    json.dump(samples, file)

print('=' * 40)
print(f'Saved final results to {args.result_path}')

success_score = np.mean([sample['success'] for sample in samples])
driving_score = np.mean([sample['metrics']['score'] for sample in samples])

print(f'Num episodes: {len(samples)}')
print(f'Average success score: {success_score}')
print(f'Average driving score: {driving_score}')
print('Done')