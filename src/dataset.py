# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load datasets used in the figures."""
import pickle
import shutil
import subprocess
import numpy as np
import requests
import tree


BUCKET_PATH = 'https://storage.googleapis.com/dm_spatial_memory_pipeline/'
FIG2_FILES = ['trajectories.pickle']
FIG3_FILES = ['hd_stats.pickle', 'parameters.pickle', 'visual_inputs.pickle']

LOCAL_PATH = 'spatial_memory_pipeline_data'

FIG2_TRAJECTORIES_PATH = f'./{LOCAL_PATH}/trajectories.pickle'
FIG3_PARAMETERS_PATH = f'./{LOCAL_PATH}/parameters.pickle'
FIG3_VISUAL_INPUTS_PATH = f'./{LOCAL_PATH}/visual_inputs.pickle'
FIG3_HD_STATS_PATH = f'./{LOCAL_PATH}/hd_stats.pickle'


def _download_files(filenames):
  subprocess.check_call(['mkdir', '-p', f'{LOCAL_PATH}'])
  for fname in filenames:
    url = f'{BUCKET_PATH}{fname}'
    dest = f'{LOCAL_PATH}/{fname}'
    print(f'Downloading: {url}')
    with requests.get(url, stream=True) as r, open(dest, 'wb') as w:
      r.raise_for_status()
      shutil.copyfileobj(r.raw, w)


def download_figure2_files():
  _download_files(FIG2_FILES)


def download_figure3_files():
  _download_files(FIG3_FILES)


def load_figure_2_trajectories():
  with open(FIG2_TRAJECTORIES_PATH, 'rb') as fh:
    trajectories = pickle.load(fh)
  return trajectories


def load_figure_3_parameters():
  with open(FIG3_PARAMETERS_PATH, 'rb') as fh:
    parameters = pickle.load(fh)
  return parameters


def load_figure_3_hd_stats():
  with open(FIG3_HD_STATS_PATH, 'rb') as fh:
    hd_stats = pickle.load(fh)
  return hd_stats


def load_figure_3_visual_inputs():
  with open(FIG3_VISUAL_INPUTS_PATH, 'rb') as fh:
    visual_inputs = pickle.load(fh)
  return visual_inputs


def flatten_trajectories(traj_ds):
  """Turn a trajectory dataset into a sample dataset, and add a time field."""
  def _flatten(x):
    """Flatten the first 2 dims of fields with > 2 dims."""
    if isinstance(x, np.ndarray) and x.ndim > 2:
      return np.reshape(x, (-1,) + x.shape[2:])
    else:
      return x
  ds = tree.map_structure(_flatten, traj_ds)
  return ds


def description(x):
  if isinstance(x, dict):
    return '\n'.join([f'{k}: {description(v)}' for k, v in sorted(x.items())])
  if isinstance(x, np.ndarray):
    return f'array{x.shape}'
  elif isinstance(x, list):
    return f'list({", ".join([description(k) for k in x])})'
