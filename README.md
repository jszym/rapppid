# RAPPPID

***R**egularised **A**utomative **P**rediction of **P**rotein-**P**rotein **I**nteractions using **D**eep Learning*

---

RAPPPID is a deep learning model for predicting protein interactions. You can 
read more about it in [our preprint](https://doi.org/10.1101/2021.08.13.456309).

## How to Use RAPPPID

### Training New Models
It's possible to train a RAPPPID model using the `train.py` utility. For precise instructions, see [docs/train.md](docs/train.md).

### Data
See [docs/data.md](docs/data.md) for information about downloading data from the manuscript, or preparing your own datasets.

### Infering
See [docs/infer.md](docs/infer.md) for advice on how to use RAPPPID for infering protein interaction probabilities.

## Environment/Requirments

The conda environment file (`environment.yml`) is available in the root of this
repository. This lists all the python libraries and the versions used for 
running RAPPPID.

You'll need an NVIDIA GPU which is CUDA compatible. RAPPPID was tested on RTX 2080, V100, and A100 GPUs.


## License

RAPPPID

***R**egularised **A**utomative **P**rediction of **P**rotein-**P**rotein **I**nteractions using **D**eep Learning*

Copyright (C) 2021  Joseph Szymborski

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.