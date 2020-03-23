# RNN-RC-Chaos

This project contains minimal implementations of RNN architectures trained with Backpropagation through time (BPTT) and Reservoir Computing (RC) for high-dimensional time-series forecasting. The following models are implemented:
- Long short-term memory **(LSTM)** trained with BPTT
- **Unitary** RNNs trained with BPTT
- Reservoir computers **(RC)** or Echo-state-networks **(ESN)**
- Deep reservoir computers **(Deep-RC)** or  **(Deep-ESN)** (private - ask for permission from pvlachas@ethz.ch)
- Multilayered perceptron (feedforward MLP) based on a windowing approach.

Moreover, spatial parallelization of the aforementioned models are implemented according to [1].

## REQUIREMENTS

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
source venv-RNN-RC-Chaos/bin/activate
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

## DATASETS

The data to run a small demo are provided in the local ./Data folder

## DEMO

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

## Note

This is only a minimal version of the code under development in the CSE-lab.
Please contact pvlachas@ethz.ch if you want to get informed, take a look at the latest version, with more features, models and capabilities.

### Relevant Publications
[1] *Backpropagation Algorithms and Reservoir Computing in Recurrent Neural Networks for the Forecasting of Complex Spatiotemporal Dynamics*, Pantelis R. Vlachas, Jaideep Pathak, Brian R. Hunt, Themistoklis P. Sapsis, Michelle Girvan, Edward Ott, Petros Koumoutsakos
Journal of Neural Networks (accepted)

[2] *Model-Free Prediction of Large Spatiotemporally Chaotic Systems from Data: A Reservoir Computing Approach*, Jaideep Pathak, Brian Hunt, Michelle Girvan, Zhixin Lu, and Edward Ott
Physical Review Letters 120 (2), 024102, 2018

[3] *Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks*, Pantelis R. Vlachas, Wonmin Byeon, Zhong Y. Wan, Themistoklis P. Sapsis and Petros Koumoutsakos
Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 474 (2213), 2018
   




