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

"""Classify units according to measureas used for biological cells."""
import numpy as np

from spatial_memory_pipeline import scores


def _make_intervals(x):
  """Given [0, 1, 2], return [-0.5, 0.5, 1.5, 2.5]."""
  d = np.diff(x)
  return np.concatenate([
      [x[0] - d[0] / 2.0],  # first interval's left bound
      x[:-1] + d / 2.0,  # interval separations
      [x[-1] + d[-1] / 2.0]  # last interval's right bound
  ])


def _rotate_distances(distances, hd):
  """Rotate allocentric distances to egocentric according to head direction.

  Args:
    distances: Array of shape [B, n_angles].`n_angles` the number of angles in
      the rat's distance sensor. The values are the distances to the closest
      wall for each world (allocentric) angle. The angles they correspond to are
      supposed to be equispaced at 2 * pi / `n_angles` rad of each other.
    hd: Array of shape [B]. For each distance, the corresponding allocentric
      head direction, in radians.

  Returns:
    Array of same shape as `distances`, each row shifted (rotated) to make
    the distances egocentric.
  Raises:
    ValueError: The dimensions of `distances` and/or `hd` are not as expected.
  """
  hd = hd.squeeze()  # it comes with trailing dimension typically
  if hd.ndim != 1:
    raise ValueError("Head direction should be a vector.")
  if distances.ndim != 2:
    raise ValueError("Distances should be a 2-D array.")
  if hd.shape[0] != distances.shape[0]:
    raise ValueError("Head direction should have same batch size as distances.")
  n_angles = distances.shape[1]
  shifts = np.mod(-np.round(n_angles * hd / (2 * np.pi)).astype(np.int),
                  n_angles)
  r_idx, c_idx = np.ogrid[:distances.shape[0], :distances.shape[1]]
  c_idx = c_idx - shifts[:, None]
  return distances[r_idx, c_idx]


def bvc_rms_egocentric(activations, world_distances, hd, distances, **kwargs):
  ego_distances = _rotate_distances(world_distances, hd)
  return bvc_rms(activations, ego_distances, distances, **kwargs)


def bvc_rms_allocentric(activations, world_distances, unused_hd, distances,
                        **kwargs):
  return bvc_rms(activations, world_distances, distances, **kwargs)


def bvc_rms(activations, world_distances, distances, for_plotting=False):
  """Compute boundary vector cell (BVC) ratemaps.

  Computes the mean activation binned by distance and (allocentric)
  angle to the nearest wall. The angular binning is the same as in the
  `world_distances` matrix. The distance binning is the one in distances.

  Args:
    activations: Array of shape [B, n_units]. `B` is the batch size, `n_units`
      the size of an the RNN layer we want to compute score for.
    world_distances: Array of shape [B, n_angles]. `B` is the same batch size as
      activations, `n_angles` the number of angles in the rat's distance sensor.
      The values are the distances to the closest wall for each world
      (allocentric) angle.
    distances: Distance binning edges.
    for_plotting: Boolean that indicates whether to have a final pi angle as it
      makes the radar plots look better.

  Returns:
    A 3-tuple of:
      distances: vector of bin centers for BVC distances.
      angles: vector of bin centers for BVC angles.
      correlations: a [n_distances, n_angles, n_units] matrix
        with the mean activations for each distance and angle bin.
  """
  n_angles = world_distances.shape[1]
  if for_plotting:
    angles = np.linspace(-np.pi, np.pi, n_angles + 1)
  else:
    angles = np.linspace(-np.pi, np.pi, n_angles + 1)[:-1]

  # Create bins of distances instead of real-values
  distance_edges = _make_intervals(distances)
  distance_bins = np.digitize(world_distances, distance_edges)

  bin_counts = np.apply_along_axis(
      np.bincount,
      0,
      distance_bins,
      weights=None,
      minlength=len(distance_edges) + 1)
  n_units = activations.shape[1]
  rms = []
  for unit_idx in range(n_units):
    rms.append(
        np.apply_along_axis(
            np.bincount,
            0,
            distance_bins,
            weights=activations[:, unit_idx],
            minlength=len(distance_edges) + 1) / (bin_counts + 1e-12))
  return distances, angles, np.dstack(rms)[1:-1, ...]


def get_resvec_shuffled_bins_threshold(ang_rmaps,
                                       percentile=99,
                                       n_shuffles=1000):
  """Get the percentile of resvec length after shuffling bins.

  Args:
    ang_rmaps: angular ratemaps.
    percentile: percentile of resvec lengths to return.
    n_shuffles: number of times to shuffle each unit.

  Returns:
    The resultant vector length threshold.
  """
  assert ang_rmaps.ndim == 2
  nbins, n_units = ang_rmaps.shape
  rv_scorer = scores.ResultantVectorHeadDirectionScorer(nbins)
  shuffle_rv_lens = np.asarray([
      rv_scorer.resultant_vector_score(
          np.random.permutation(ang_rmaps[:, i % n_units]))
      for i in range(n_units * n_shuffles)
  ]).flatten()
  return np.percentile(shuffle_rv_lens, percentile)


def get_hd_shuffled_bins_threshold(activations,
                                   hds,
                                   n_bins=20,
                                   percentile=99,
                                   n_shuffles=1000):
  """Calculate the resvec length threshold for HD cells.

  Args:
    activations: array of cells activations.
    hds: head direction of the agent.
    n_bins: number of head-direction bins to use.
    percentile: percentile of shuffled resvec lengths to return.
    n_shuffles: number of bin shuffles to use.

  Returns:
    The resultant vector length threshold.
  """
  rv_scorer = scores.ResultantVectorHeadDirectionScorer(n_bins)
  n_units = activations.shape[-1]
  ang_rmaps = np.asarray([
      rv_scorer.calculate_hd_ratemap(hds[:, 0], activations[:, i])
      for i in range(n_units)
  ]).T
  return get_resvec_shuffled_bins_threshold(
      ang_rmaps, percentile=percentile, n_shuffles=n_shuffles)


def get_hd_cell_parameters(acts, hds, threshold, n_bins=20):
  """Calculate which cells are head-direction and their preferred angle.

  Args:
    acts: array of cells activations.
    hds: head direction of the agent.
    threshold: resvec length at which a unit is considered head-direciton cell.
    n_bins: number of head-direction bins to use.

  Returns:
    Array of bools indicating which cells are HD cells, Array of resvec lengths,
    and Array of preferred head-direction angles.
  """
  rv_scorer = scores.ResultantVectorHeadDirectionScorer(n_bins)
  n_units = acts.shape[-1]
  ang_rmaps = [rv_scorer.calculate_hd_ratemap(hds[:, 0], acts[:, i])
               for i in range(n_units)]
  rv_lens = np.asarray([rv_scorer.resultant_vector_score(ang_rmaps[i])
                        for i in range(n_units)]).flatten()
  rv_angles = np.asarray([rv_scorer.resultant_vector_angle(ang_rmaps[i])
                          for i in range(n_units)]).flatten()
  return rv_lens > threshold, rv_lens, rv_angles


def get_bvc_cell_parameters(rms_dists,
                            rms_angs,
                            rms_vals,
                            threshold,
                            unused_distance_step=2.5):
  """Calculate which cells are bvc and their preferred distance and angle.

  Args:
    rms_dists: Distances in the rms data.
    rms_angs: Angles in the rms data.
    rms_vals: Array of activity of shape [n_distances, n_angles, n_units].
    threshold: resultant vector length threshold to consider a cell bvc.
    unused_distance_step: unused parameter.

  Returns:
    Array of bools indicating which cells are BVC cells, Array of preferred
    angles, Array of preferred distances, and array of resvec lengths.
  """
  _, n_angles, n_units = rms_vals.shape
  l = rms_vals.sum(axis=0)
  y = np.sum(np.sin(rms_angs)[:, np.newaxis] * l, axis=0)
  x = np.sum(np.cos(rms_angs)[:, np.newaxis] * l, axis=0)

  # In order to make this consistent with the ResVec scorer used for the HD
  # cells, use arctan(x, y).
  # This makes the normal angle be pi/2 - angle returned by arctan
  bvc_angles = np.arctan2(x, y)

  bvc_orig_angles = np.arctan2(y, x)
  bvcbins = (
      np.digitize(bvc_orig_angles, np.concatenate([rms_angs, [np.pi]])) - 1)
  bvc_dists = rms_dists[np.argmax(
      rms_vals[:, bvcbins, range(len(bvcbins))], axis=0)]
  rv_scorer = scores.ResultantVectorHeadDirectionScorer(n_angles)
  rvls = [rv_scorer.resultant_vector_score(rms_vals[..., i].sum(axis=0))[0]
          for i in range(n_units)]
  rvls = np.asarray(rvls)
  is_bvc = (rvls > threshold)
  return is_bvc, bvc_angles, bvc_dists, rvls


def classify_units_into_representations(activities, hds, pos, wall_distances,
                                        n_shuffles=1000,
                                        distance_bin_size=0.025,
                                        percentile=99):
  """Identify which cells are HD, egoBVC or alloBVC (non-exclusive).

  Args:
    activities: array of cells activations.
    hds: array head-direction of the agent.
    pos: array of position of the agent in the enclosure.
    wall_distances: array of distances to walls at each time-step.
    n_shuffles: number of shuffles used in calculating the thresholds.
    distance_bin_size: size of distance bins.
    percentile: percentile of score in the shuffled data to use as threshold.

  Returns:
    Tuple of
    (whether each cell is a head direction cell,
     head-direction resvec length threshold,
     hd score of each cell,
     head-direction resultant-vector length of each cell,
     preferred head direction of each cell,
     whether each cell is an egocentric boundary cell,
     ego-bvc resvec threshold,
     ego-bcv score for each cell,
     preferred egocentric boundary distance to wall of each cell,
     preferred egocentric boundary angle to wall of each cell,
     whether each cell is an allocentric boundary cell,
     ego-bvc resvec threshold,
     ego-bcv score for each cell,
     preferred allocentric boundary distance to wall of each cell,
     preferred allocentric boundary angle to wall of each cell)
  """
  # Calculate preferred wall distances up to half of the enclosure size
  max_d = (pos[:, 0].max() - pos[:, 0].min()) / 2
  distances = np.arange(distance_bin_size, max_d, distance_bin_size)

  # Identify head-direction cells
  hd_threshold = get_hd_shuffled_bins_threshold(
      activities, hds, percentile=percentile, n_shuffles=n_shuffles)
  is_hd, hd_rv_lens, hd_rv_angles = get_hd_cell_parameters(
      activities, hds, hd_threshold, n_bins=20)
  hd_score = hd_rv_lens

  # Identify egocentric-boundary cells
  erms_dists, erms_angs, erms_vals = bvc_rms_egocentric(
      activities, wall_distances, hds, distances)
  ego_threshold = get_resvec_shuffled_bins_threshold(
      erms_vals.sum(axis=0), percentile=percentile, n_shuffles=n_shuffles)
  is_ego, ego_angle, ego_dist, ego_score = get_bvc_cell_parameters(
      erms_dists, erms_angs, erms_vals, ego_threshold)

  # Identify allocentric boundary cells
  arms_dists, arms_angs, arms_vals = bvc_rms_allocentric(
      activities, wall_distances, hds, distances)
  allo_threshold = get_resvec_shuffled_bins_threshold(
      arms_vals.sum(axis=0), percentile=percentile, n_shuffles=n_shuffles)
  is_allo, allo_angle, allo_dist, allo_score = get_bvc_cell_parameters(
      arms_dists, arms_angs, arms_vals, allo_threshold)

  return (is_hd, hd_threshold, hd_score, hd_rv_lens, hd_rv_angles,
          is_ego, ego_threshold, ego_score, ego_dist, ego_angle,
          is_allo, allo_threshold, allo_score, allo_dist, allo_angle)
