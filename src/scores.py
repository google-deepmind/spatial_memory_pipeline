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

"""Cell-type score calculations."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.spatial
import scipy.stats


def positional_correlations_manipulation(pos, hds, acts, pos_b, hds_b, acts_b):
  """Positional correlation between two datasets."""
  pos_corr = list()
  for unit in range(acts.shape[1]):
    ratemap_a = calculate_ratemap(
        pos[:, 0], pos[:, 1], acts[:, unit],
        hd=hds[:])[0][0].reshape([1, -1])
    ratemap_b = calculate_ratemap(
        pos_b[:, 0], pos_b[:, 1], acts_b[:, unit],
        hd=hds_b[:])[0][0].reshape([1, -1])
    not_nan = np.logical_and(np.isfinite(ratemap_a), np.isfinite(ratemap_b))
    cc = np.corrcoef(ratemap_a[not_nan], ratemap_b[not_nan])[0, 1]
    pos_corr.append(cc)
  return np.asarray(pos_corr)


def positional_correlations(pos, hds, acts):
  """Spatial stability as the correlation between two halves of data."""
  half = pos.shape[0] // 2
  return positional_correlations_manipulation(pos[:half, ...],
                                              hds[:half],
                                              acts[:half, ...],
                                              pos[half:, ...],
                                              hds[half:],
                                              acts[half:, ...])


def calculate_ratemap(xs,
                      ys,
                      activations,
                      bin_size=0.14,  # to divide 2.2 meters into 16 bins
                      coords_range=None,
                      statistic="mean",
                      hd=None):
  """Calculate 2D arrays of activations binned by spatial location.

  Args:
    xs: vector of length B with the x position of the agent.
    ys: vector of length B with the y position of the agent.
    activations: vector of length B with the activation of one cell at each
      position.
    bin_size: In distance units, for binning across both dimensions.
    coords_range: A pair of tuples ((xmin, xmax), (ymin, ymax)). If not given,
      the `xs` and `ys` ranges are used.
    statistic: What to compute over activations in each bin.
    hd: Optional. The head directions corresponding to each position.
      If given, not one but 9 ratemaps will be computed: the general
      one and one for each head-direction octant.
  Returns:
    A 3-tuple:
      The first element is the 2d ratemap if `hd` is None. If `hd`
        is given, it will be a 2-tuple, where the first one is the
        normal ratemap and second is a H x W x 8 per-octant ratemap, octants
        being [-pi/8, pi/8], [pi/8, 3pi/8], [3pi/8, 5pi/8], [5pi/8, 7pi/8],
        [-7pi/8, -7pi/8], [-7pi/8, -5pi/8], [-5pi/8, -3pi/8], and
        [-3pi/8, -pi/8].
      The second and third elements are the x and y edges used for binning.
  """
  def _get_edges(min_v, max_v):
    v_range = max_v - min_v
    n_bins = np.ceil(v_range / bin_size).astype(int)
    pad = (n_bins * bin_size - v_range) / 2
    return np.linspace(min_v - pad / 2, max_v + pad / 2, n_bins + 1)

  if coords_range is None:
    coords_range = ((xs.min(), xs.max()), (ys.min(), ys.max()))
  x_edges = _get_edges(coords_range[0][0], coords_range[0][1])
  y_edges = _get_edges(coords_range[1][0], coords_range[1][1])
  rms = scipy.stats.binned_statistic_2d(
      xs, ys, activations, bins=(x_edges, y_edges), statistic=statistic)[0]
  if hd is not None:
    octants = np.mod(np.round(np.squeeze(hd) / (np.pi / 4)), 8)
    octant_edges = np.linspace(-0.5, 7.5, 9)
    octant_rms = scipy.stats.binned_statistic_dd(
        (xs, ys, octants), activations,
        bins=(x_edges, y_edges, octant_edges),
        statistic=statistic)[0]
    rms = (rms, octant_rms)

  return rms, x_edges, y_edges


class ResultantVectorHeadDirectionScorer(object):
  """Class for scoring correlation between activation and head-direction."""

  def __init__(self, nbins):
    """Constructor.

    Args:
      nbins: Number of bins for the angle.
    """
    self._bins = np.linspace(-np.pi, np.pi, nbins + 1)
    self._bin_centers = 0.5 * self._bins[:-1] + 0.5 * self._bins[1:]
    self._bin_vectors = np.asarray(
        [np.cos(self._bin_centers),
         np.sin(self._bin_centers)]).T  # n_bins x 2

  def calculate_hd_ratemap(self, hds, activations, statistic="mean"):
    hds = np.arctan2(np.sin(hds), np.cos(hds))
    total_bin_act = scipy.stats.binned_statistic(
        hds,
        activations,
        bins=self._bins,
        statistic=statistic,
        range=(-np.pi, np.pi))[0]
    return total_bin_act

  def resultant_vector(self, ratemaps):
    if ratemaps.ndim == 1:
      ratemaps = ratemaps[np.newaxis, :]
    resv = (np.matmul(ratemaps, self._bin_vectors) /
            (np.sum(ratemaps, axis=1)[:, np.newaxis] + 1e-5))
    return resv

  def resultant_vector_score(self, ratemaps):
    resv = self.resultant_vector(ratemaps)
    return np.hypot(resv[:, 0], resv[:, 1])

  def resultant_vector_angle(self, ratemaps):
    resv = self.resultant_vector(ratemaps)
    return np.arctan2(resv[:, 0], resv[:, 1])

  def plot_polarplot(self,
                     ratemaps,
                     ax=None,
                     title=None,
                     relative=False,
                     positive_lobe_color="b",
                     negative_lobe_color="r",
                     **kwargs):
    """ratemaps: list of ratemaps."""
    if isinstance(ratemaps, list):
      ratemaps = np.asarray(ratemaps, dtype=np.float32)
    if ax is None:
      ax = plt.gca()
    # Plot the ratemap
    for i in range(ratemaps.shape[0]):
      ang = np.append(self._bin_centers, [self._bin_centers[0]], axis=0)
      pos_rm = ratemaps[i] * (ratemaps[i] > 0)
      pos_lens = np.append(pos_rm, [pos_rm[0]], axis=0)
      neg_rm = -ratemaps[i] * (ratemaps[i] < 0)
      neg_lens = np.append(neg_rm, [neg_rm[0]], axis=0)
      if relative:
        pos_lens /= pos_lens.sum()
        neg_lens /= neg_lens.sum()
      # Our ratemap convention is first coordinate positive up, second
      # coordinate positive right. To have the same orientation in polar
      # plots we do pi/2 - angle
      ax.plot(np.pi / 2 - ang, pos_lens, color=positive_lobe_color, **kwargs)
      ax.plot(np.pi / 2 - ang, neg_lens, color=negative_lobe_color, **kwargs)
    if title is not None:
      ax.set_title(title)
