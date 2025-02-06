# MMCLKin
<div id="top" align="center">
 <h3>Enhancing Kinase-Inhibitor Activity and Selectivity Prediction Through Multimodal and Multiscale Contrastive Learning with Attention Consistency<h3>

![MMCLKin](https://github.com/Yanara-Tian/MMCLKin/blob/main/Framework%20of%20MMCLKin.png)


## OS Requirements
This repository has been tested on **Linux**  operating systems.

## Python Dependencies
* Python (version >= 3.8.15) 
* Torch (version >= 1.12.1)
* Torchvision (version>=0.13.1)
* Torchaudio (version>=0.12.1)
* matplotlib (version >= 3.7.5)
* Bio (version >= 1.6.2)
* scipy (version >= 1.10.1)
* RDKit (version >= 2024)
* networkx (version >= 2.7.1)
* pyg_lib (version >= 0.4.0)
* torch_geometric (version >= 2.6.1)
* torch_scatter (version >= 2.1.2)
* torch_sparse (version >= 0.6.18)
* torch_cluster (version >= 1.6.3)
* torch_spline_conv (version >= 1.2.2)
* dgl-cu116

## Installation Guide
Create a virtual environment to run the code of MMCLKin.<br>
It is recommended to use conda to manage the virtual environment.The installation method for conda can be found [here](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html#installing-on-linux).<br>
Make sure to install torch with the cuda version that fits your device.<br>
This process usually takes few munites to complete.<br>
```
git clone https://github.com/Yanara-Tian/MMCLKin.git
cd MMCLKin
export PYTHONPATH=$PWD:$PYTHONPATH
```
## Dependencies
This package is tested with Python 3.8.15 and CUDA 11.0 on Ubuntu 20.04. Run the following to create a conda environment and install the required Python packages (modify `torch+cu11.6` according to your CUDA version). 
```bash
conda create -n mmclk python=3.8.15
conda activate mmclk
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --pre dgl-cu116 -f https://data.dgl.ai/wheels-test/repo.html
pip install matplotlib
pip install rdkit
pip install scipy
pip install Bio
pip install transformers
pip install pyg_lib torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
pip install networkx==2.7.1
```
Running the above lines of `pip install` should be sufficient to install all  MMCLKin's required packages (and their dependencies). Specific versions of the packages we tested were listed in `requirements.txt`.

## Contact
Please submit GitHub issues or contact Yanan Tian(yanan.tian@mpu.edu.mo) for any questions related to the source code.
