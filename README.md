# MMCLKin
<div id="top" align="center">
 <h3>Enhancing Kinase-Inhibitor Activity and Selectivity Prediction Through Multimodal and Multiscale Contrastive Learning with Attention Consistency<h3>
</div>
  ![MMCLKin](https://github.com/Yanara-Tian/MMCLKin/blob/main/Framework%20of%20MMCLKin.png)

## OS Requirements
This repository has been tested on **Linux**  operating systems.

## Python Dependencies
* Python (version >= 3.8.15) 
* Torch (version >= 1.12.1)
* Torchvision (version>=0.13.1)
* Torchaudio (version>=0.12.1)
* dgl-cu116
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

## Installation Guide
Create a virtual environment to run the code of MMCLKin.<br>
It is recommended to use conda to manage the virtual environment.The installation method for conda can be found [here](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html#installing-on-linux).<br>
Make sure to install torch with the cuda version that fits your device.<br>
This process usually takes few munites to complete.<br>
```
git clone https://github.com/Yanara_Tian/MMCLKin.git
cd MMCLKin
chmod +x ./setup_MMCLKin.sh
./setup_MMCLKin.sh
conda activate mmclkin_env
```
