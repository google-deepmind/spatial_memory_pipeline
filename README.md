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

* [Figure 2](https://colab.research.google.com/github/deepmind/spatial_memory_pipeline/blob/master/figure_2.ipynb): displaying the types and distribution of the
  spatial representations. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/spatial_memory_pipeline/blob/master/figure_2.ipynb)

* [Figure 3](https://colab.research.google.com/github/deepmind/spatial_memory_pipeline/blob/master/figure_3.ipynb): analyzing the trained head-direction cell network. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/spatial_memory_pipeline/blob/master/figure_3.ipynb)

## References

[1] Benigno Uria, Borja Ibarz, Andrea Banino, Vinicius Zambaldi,
Dharshan Kumaran, Demis Hassabis, Caswell Barry, Charles Blundell.
*The Spatial Memory Pipeline: a model of egocentric to allocentric
understanding in mammalian brains*. [bioRxiv 2020.11.11.378141](https://www.biorxiv.org/content/10.1101/2020.11.11.378141v1.full)

## Disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the License. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY).  You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.
