# MMCLKin
<div id="top" align="center">
 <h3>Enhancing Kinase-Inhibitor Activity and Selectivity Prediction Through Multimodal and Multiscale Contrastive Learning with Attention Consistency<h3>
</div>
![MMCLKin](https://github.com/Yanara-Tian/MMCLKin/blob/main/Framework%20of%20MMCLKin.png)

## OS Requirements
This repository has been tested on **Linux**  operating systems.

## Python Dependencies
* Python (version >= 3.8) 
* PyTorch (version >= 1.12.1) 
* RDKit (version >= 2019)
* TorchDrug (version == 0.2.1)
* fair-esm (version == 2.0.1)
* Py3Dmol (version ==2.0.3)

## Installation Guide
Create a virtual environment to run the code of EasIFA.<br>
It is recommended to use conda to manage the virtual environment.The installation method for conda can be found [here](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html#installing-on-linux).<br>
Make sure to install pytorch with the cuda version that fits your device.<br>
This process usually takes few munites to complete.<br>
```
git clone https://github.com/wangxr0526/EasIFA.git
cd EasIFA
chmod +x ./setup_EasIFA.sh
./setup_EasIFA.sh
conda activate easifa_env
```
