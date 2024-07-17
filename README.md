# Code for Holodepth: Programmable Depth-Varying Projection via Computer-Generated Holography

`main_depth_variation.py` contains sample code for comparing the level of depth variation between a projector with etendue expanded by a static optic, and a naive projector magnified to match the same field-of-view.

`main_opt.py` contains a sample training loop for calibrating a projector with etendue expanded by a static optic with the model presented in Eq. (8) of the paper.

The wavefront propagation code is adapted from [Holotorch](https://github.com/facebookresearch/holotorch/), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)