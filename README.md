# The Spatial Memory Pipeline

The Spatial Memory Pipeline [1] is a model that learns to map egocentric visual
and velocity information into allocentric spatial representations. The
representations learned by the model resemble head-direction, boundary-vector
and place-cell responses found in the mammalian hippocampal formation.

This directory contains notebooks that reproduce figures 2 and 3 in [1] based
on the inputs, outputs and parameters of the trained model. These figures
illustrate the types of representations learned by the model, as well as
the properties of the learned head-direction network.

## Usage

We provide one Colab notebook to reproduce each of figures 2 and 3 of [1]:

* [Figure 2](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/spatial_memory_pipeline/figure_2.ipynb): displaying the types and distribution of the
  spatial representations. [![Open In Colab] (https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/spatial_memory_pipeline/figure_2.ipynb)

* [Figure 3](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/spatial_memory_pipeline/figure_3.ipynb): analyzing the trained head-direction cell network. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/spatial_memory_pipeline/figure_3.ipynb)

## References

[1] Benigno Uria, Borja Ibarz, Andrea Banino, Vinicius Zambaldi,
Dharshan Kumaran, Demis Hassabis, Caswell Barry, Charles Blundell.
*The Spatial Memory Pipeline: a model of egocentric to allocentric
understanding in mammalian brains*. [bioRxiv 2020.11.11.378141](https://www.biorxiv.org/content/10.1101/2020.11.11.378141v1.full)

## Disclaimer

This is not an official Google product.
