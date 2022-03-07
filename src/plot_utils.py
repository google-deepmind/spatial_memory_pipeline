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

"""Plotting utils."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def octant_grid(rms, rms_octants):
  """Put main rms and the 8 octant rmss in a single figure.

  Args:
    rms: H x W array.
    rms_octants: H x W x 8 array, assuming the i-th 3D slice corresponds to the
      i-th octant (0=octant centered at 0 degrees, 1 octant centered at 45
      degrees, etc.)

  Returns:
    (3H + padding) x (3W + padding) array with the individual rms in correct
    order to be plotted.
  """

  all_rms = np.dstack((rms_octants, rms))
  padded_rms = np.zeros((all_rms.shape[0] + 2, all_rms.shape[1] + 2, 9))
  padded_rms[1:-1, 1:-1, :] = all_rms
  h, w = padded_rms.shape[:2]
  display_order = [[5, 4, 3], [6, 8, 2], [7, 0, 1]]
  grid = padded_rms[..., display_order]  # H x W x 3 x 3
  grid = np.rollaxis(grid, 2, 0)  # 3 x H x W x 3
  grid = np.rollaxis(grid, 3, 2)  # 3 x H x 3 x W
  return grid.reshape(3 * h, 3 * w)


def setup_spines(ax, lw=1.5):
  for sp in ax.spines.values():
    sp.set_linewidth(lw)
    sp.set_color("#333333")


def imshow(ax, image):
  ax.imshow(image, origin="lower", interpolation="none")
  ax.set_axis_off()


def prop_hist(ax, data, bins, polar=False, **kwargs):
  hist = np.histogram(data, bins=bins)[0]
  hist = hist / len(data)
  # For polar plots we want the *area* to be determined
  # by the proportion, not the radius
  if polar:
    hist = np.sqrt(hist)
  w = (bins.max() - bins.min()) / (len(bins) - 1)
  ax.bar(bins[:-1] + w/2, hist, width=w, **kwargs)


def octant_ratemap_to_rgb(rms, per_octant, min_perc=0, max_perc=100,
                          cmap="jet"):
  """Produce an rgb image from per-octant ratemaps.

  Normally we would let plt.imshow normalize values automatically when
  it applies the colormap, but since we can have nans and we may want to
  make a grid of octants (where the values between the octants would be
  meaninglessly color-mapped) we have to convert the ratemap to rgb color
  before we do imshow, so we normalize it manually.

  Args:
    rms: a 2-tuple with (H x W, H x W x 8) global and per-octant ratemaps.
    per_octant: (bool) If True, we'll return a per-octant rgb grid, otherwise
      just the normalized RGB global ratemap.
    min_perc: (float, default 0) Percentile of the range of ratemap values
      to use as the bottom of the color scale.
    max_perc: (float, default 100) Percentile of the range of ratemap values
      to use as the top of the color scale.
    cmap: (str) Name of the colormap to use.
  Returns:
    A single RGB image with the ratemap or the 9 per-octant + global grid.
  """
  data = np.ravel(rms[0])
  if per_octant:
    data = np.concatenate((data, np.ravel(rms[1])))
  lims = np.nanpercentile(data, [min_perc, max_perc])
  center_rms = (rms[0] - lims[0]) / (lims[1] - lims[0] + 1e-16)
  the_rims = None
  if per_octant:
    side_rms = (rms[1] - lims[0]) / (lims[1] - lims[0] + 1e-16)
    center_rms = octant_grid(center_rms + 1, side_rms + 1)
    the_rims = center_rms == 0
    center_rms -= 1
    center_rms[the_rims] = 0

  rms = center_rms
  nans = np.isnan(rms)
  rms[nans] = lims[0]
  rms = matplotlib.cm.get_cmap(cmap)(rms)[..., :3]  # to rgb, discard alpha
  rms[nans] = 1.0
  if the_rims is not None:
    rms[the_rims] = 1.0
  return rms


class SuperFigure(object):
  """Plotting figures with lettered subpanels."""

  def __init__(self, width_inches, height_inches, dpi=200, **kwargs):
    self._dpi = dpi
    self._wi = width_inches
    self._hi = height_inches
    kwargs["figsize"] = (width_inches * dpi / 72, height_inches * dpi / 72)
    kwargs["dpi"] = 72
    self._fig, self._allax = plt.subplots(1, 1, **kwargs)
    self._allax.set_axis_off()
    self.label_letter = "A"
    self.label_suffix = ")"

  def next_label(self):
    curr = self.label_letter
    self.label_letter = chr(ord(self.label_letter) + 1)
    return curr + self.label_suffix

  def add_subplots_to_figure(self, n_rows, n_cols, pos_x, pos_y, width, height,
                             hspace=0.1, vspace=0.1, text=None, text_offset=0.0,
                             **subplot_kw):
    """Add a grid of subplots.

    Args:
      n_rows: Number of rows of subplots to add.
      n_cols: Number of columns of subplots to add.
      pos_x: x position in the figure of the top-left subplot.
      pos_y: y position in the figure of the top-left subplot.
      width: width of each subplot.
      height: height of each subplot.
      hspace: horizontal space between subplot columns.
      vspace: vertical space between subplot rows.
      text: text next to the top-left subplot for referring to the panel.
      text_offset: x offset of the text for fine-tuning the look.
      **subplot_kw: dictionary of extra parameters in subplot creation.

    Returns:
      An array of matplotlib axes.
    """
    try:
      text_offset = np.asarray(text_offset).reshape([2])
    except ValueError:
      text_offset = np.asarray([text_offset, 0]).reshape([2])
    if text is not None:
      self.add_text(pos_x + text_offset[0], pos_y + text_offset[1], text,
                    fontsize=9)

    size_inches = self._fig.get_size_inches() / (self._dpi / 72.0)
    # Inches to relative
    pos_x /= size_inches[0]
    pos_y /= size_inches[1]
    width /= size_inches[0]
    height /= size_inches[1]
    hspace /= size_inches[0]
    vspace /= size_inches[1]

    # Use distance in inches from the top, not from the bottom
    pos_y = 1 - (pos_y + height * n_rows + vspace * (n_rows - 1))

    axs = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
      for col in range(n_cols):
        axs[row, col] = self._fig.add_axes((pos_x + col * (hspace + width),
                                            pos_y - row * (vspace + height),
                                            width, height), **subplot_kw)
    return axs

  def add_text(self, pos_x, pos_y, text, fontsize=9):
    if text == "auto":
      text = self.next_label()
    size_inches = self._fig.get_size_inches() / (self._dpi / 72.0)
    pos_x /= size_inches[0]
    pos_y /= size_inches[1]
    pos_y = 1 - pos_y
    self._fig.text(
        pos_x,
        pos_y,
        text,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=fontsize * self._dpi / 72)

  def show(self):
    return self._fig

