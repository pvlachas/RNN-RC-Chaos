# RNN-RC-Chaos

This project contains minimal implementations of RNN architectures trained with Backpropagation through time (BPTT) and Reservoir Computing (RC) for high-dimensional time-series forecasting. The following models are implemented:
- Long short-term memory **(LSTM)** trained with BPTT
- **Unitary** RNNs trained with BPTT
- Reservoir computers **(RC)** or Echo-state-networks **(ESN)**
- Multilayered perceptron (feedforward MLP) based on a windowing approach.

Moreover, spatial parallelization of the aforementioned models are implemented according to [1].
For implementation of the method to compute the Lyapunov spectrum, please refer to the repository [RNN-Lyapunov-Spectrum](https://github.com/pvlachas/RNN-Lyapunov-Spectrum).

## Code Requirements

The code requires python 3.7.3, tensorflow 1.11.0
Other required packages are: matplotlib, sklearn, psutil.
- python 3.7.3
- tensorflow 1.11.0
- matplotlib, sklearn, psutil
- mpi4py (parallel implementations)

The packages can be installed as follows: you can create a virtual environment in Python3 with:
```
python3 -m venv venv-RNN-RC-Chaos

```
Then activate the virtual environment:
```
source ./venv-RNN-RC-Chaos/bin/activate
```
Install a version of tensorflow (paper was compiled with version 1.11, which may no longer be available), here we also tested a more recent verion 1.14, (apart from warnings the code should run fine):
```
pip install tensorflow==1.14.0
```
Install the rest of the required packages with:
```
pip3 install matplotlib sklearn psutil mpi4py
```
The code is ready to run, you can test the following demo.


## Parallel architectures

Parallelized networks that take advantage of the local interactions in the state space employ MPI communication.
After installing an MPI library (open-mpi or mpich), the mpi4py library can be installed with:
```
pip3 install mpi4py
```


## Virtual environment used in the paper

The code to get the exact environment (no mpi4py/parallel models yet) used in the paper is:
```
pip install virtualenv
virtualenv venv-RNN-RC-Chaos --python=python3.7.3
source ./venv-RNN-RC-Chaos/bin/activate
pip3 install -r requirements.txt
```
In macOs to install mpi4py:
```
source ./venv-RNN-RC-Chaos/bin/activate
pushd /tmp
rm -f tmp.c && touch tmp.c
xcrun -sdk macosx clang -arch x86_64 -c tmp.c
export OMPI_CC=xcrun
export MPICH_CC=xcrun
pip install --no-cache-dir mpi4py
```


## Datasets

The data to run a small demo are provided in the local ./Data folder


## Demo

In order to run the demo in a local cluster, you can navigate to the Experiments folder, and select your desired application, e.g. Lorenz3D. There are scripts for each model. For example, you can ran a Reservoir Computer (also called Echo state network) with the following commands:
```
cd ./Experiments/Lorenz3D/Local
bash 01_ESN_auto.sh.sh
```
A statefull GRU or a parallel ESN can be run with:
```
bash 04_RNNStatefull_GRU.sh
bash 05_ESN_Parallel.sh
```
After running the command, you will see at the terminal output the training/testing progress.
You can then navigate to the folder ./Results/Lorenz3D and check the different outputs of each model.


## Contact info

This code was developed in the [CSE-lab](https://www.cse-lab.ethz.ch).
For questions or to get in contact please refer to pvlachas@ethz.ch.

## Acknowledgments

This is joint work with:
- Jaideep Pathak ([website](http://physics.umd.edu/~jpathak/), [scholar](https://scholar.google.com/citations?user=cevw0gkAAAAJ&hl=en)) 
- Brian R. Hunt ([website](http://www.math.umd.edu/~bhunt/), [scholar](https://scholar.google.com/citations?user=ten7UlMAAAAJ&hl=en))
- Themis Sapsis ([website](http://sandlab.mit.edu/), [scholar](https://scholar.google.com/citations?user=QSPXIAQAAAAJ&hl=en))
- Michelle Girvan ([website](https://sites.google.com/umd.edu/networks/home), [scholar](https://scholar.google.com/citations?user=npKBI-oAAAAJ&hl=el)) 
- Edward Ott ([website](https://umdphysics.umd.edu/people/faculty/current/item/380-edott.html), [scholar](https://scholar.google.com/citations?user=z7boxkkAAAAJ&hl=en))
- Petros Koumoutsakos ([website](https://www.cse-lab.ethz.ch/member/petros-koumoutsakos/), [scholar](https://scholar.google.ch/citations?user=IaDP3mkAAAAJ&hl=el&oi=ao)) 

## Relevant Publications
[1] P.R. Vlachas, J. Pathak, B.R. Hunt et al., *Backpropagation algorithms and
Reservoir Computing in Recurrent Neural Networks for the forecasting of complex spatiotemporal
dynamics.*
Neural Networks, 2020 (doi: https://doi.org/10.1016/j.neunet.2020.02.016.)

[2] J. Pathak, B.R. Hunt, M. Girvan, Z. Lu, and E. Ott, *Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach.*
Physical Review Letters 120 (2), 024102, 2018

[3] P.R. Vlachas, W. Byeon, Z.Y. Wan, T.P. Sapsis, and P. Koumoutsakos *Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks.*
Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 474 (2213), 2018





