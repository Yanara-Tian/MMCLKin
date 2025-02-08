# MMCLKin
<div id="top" align="center">
 <h3>Enhancing Kinase-Inhibitor Activity and Selectivity Prediction Through Multimodal and Multiscale Contrastive Learning with Attention Consistency<h3>
 </div>

![MMCLKin](https://github.com/Yanara-Tian/MMCLKin/blob/main/Framework%20of%20MMCLKin.png)

## OS Requirements
This repository has been tested on **Linux**  operating systems.

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
## Python Dependencies
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
pip install sympy
pip install scikit-learn
pip install lifelines
```
Running the above lines of `pip install` should be sufficient to install all  MMCLKin's required packages (and their dependencies). Specific versions of the packages we tested were listed in `requirements.txt`.

## Reproduce Results

### Kinase-inhibitor binding affinity 

#### Predictive performance of kinase-inhibitor binding affinity on the 3DKDavis dataset
**[1]** Download checkpoints(~298MB) and dataset(~15GB), and then extract the dataset.
```
wget pkls/3dkdavis_new_kinase_affinity/MMCLKin_DTI_mse_best.pkl
wget test_datasets/3dkdavis_new_kinase_affinity.tar.gz
cd test_datasets
tar zxvf 3dkdavis_new_kinase_affinity.tar.gz
```
**[2]** Test the prediction performance of MMCLKin for kinase-inhibitor binding affinity on 3DKDavis dataset. We offer dataset splits based on drug cold-start, kinase cold-start, or kinase-drug cold-start, ensuring that the model is tested on unseen kinases, unseen drugs, or both. To enable this option, set the --label argument to new_kinase, new_drug, or both_new. For example, to test on unseen kinases, run the following script:
```
python test_3dkdavis_affinity.py
```
#### Predictive performance of kinase-inhibitor binding affinity on the low sequence similarity dataset of 3DKKIBA
**[1]** Download checkpoints(~300MB) and dataset(~6.8GB), and then extract the dataset.
```
wget pkls/3dkkiba_new_kinase_affinity/MMCLKin_DTI_mse_best.pkl
wget test_datasets/3dkkiba_new_kinase_affinity.tar.gz
cd test_datasets
tar zxvf 3dkkiba_new_kinase_affinity.tar.gz
```
**[2]** Test the prediction performance of MMCLKin for kinase-inhibitor binding affinity on low sequence similarity dataset of 3DKDavis. For example, to evaluate on unseen kinases, run the following script:
```
python test_3dkkiba_affinity.py
```

### The selectivity of kinase inhibitors on human kinome

#### Predictive performance of the selectivity of kinase inhibitors on the 3DKDavis dataset
**[1]** Download checkpoints(~300MB) and dataset(~15GB), and then extract the dataset.
```
wget pkls/3dkkiba_new_kinase_affinity/MMCLKin_DTI_mse_best.pkl
wget test_datasets/3dkdavis_new_kinase_selectivity.tar.gz
cd test_datasets
tar zxvf 3dkdavis_new_kinase_selectivity.tar.gz
```
**[2]** To ensure comprehensive learning of human kinases, the predictive performance for kinase inhibitor selectivity of MMCLKin was evalueated under the drug cold-start splitting strategy.
```
python test_3dkdavis_selectivity.py
```
#### Predictive performance of the selectivity of kinase inhibitor on the low sequence similarity dataset of 3DKKIBA
**[1]** Download checkpoints(~300MB) and dataset(~6.1GB), and then extract the dataset.
```
wget pkls/3dkkiba_selectivity/MMCLKin_DTI_pearson_best.pkl
wget test_datasets/3dkkiba_new_kinase_selectivity.tar.gz
cd test_datasets
tar zxvf 3dkkiba_new_kinase_selectivity.tar.gz
```
**[2]** Test the predictive performance for kinase inhibitor selectivity of MMCLKin on the low sequence similarity dataset of 3DKKIBA.
```
python test_3dkkiba_selectivity.py
```
### virtual screening on the experimental structure
```
examples/virtual_screening_lrrk2.ipynb
```
### virtual screening on the unresolved crystal structure
```
examples/virtual_screening_nuak2.ipynb
```

## Feature extraction, training, and testing pipeline

### 3DKDavis 
**[1]** Feature extraction, encompassing the biochemical and conformational characteristics of kinase inhibitors, evolutionary information and the intricate spatial structural features of binding pockets and kinase domains.这里，为了更加快速的得到每个复合物体系的特征，我们首先建议您下载我们生成的所有的激酶，所有结合口袋以及所有激酶抑制剂的特征。然后执行以下命令：
```
python process_3dkdavis.py
```
**[2]** 训练和测试模型在激酶-抑制剂结合亲和力的预测性能，我们提供了三种分割方式，kinase cold-start, drug cold-start and kinase-drug cold-start, such that the model is tested on unseen proteins, unseen drugs or both. To use this option, set the argument --split_method using drug or both for the --split_method method, For example, to test on unseen drugs, run the following script.
```
python train_3dkdavis.py
```
### 3DKKIBA
**[1]** Feature extraction, encompassing the biochemical and conformational characteristics of kinase inhibitors, evolutionary information and the intricate spatial structural features of binding pockets and kinase domains.这里，为了更加快速的得到每个复合物体系的特征，我们首先建议您下载我们生成的所有的激酶，所有结合口袋以及所有激酶抑制剂的特征。然后执行以下命令：
```
python process_3dkkiba.py
```
**[2]** 训练和测试模型在激酶-抑制剂结合亲和力的预测性能，我们提供了三种分割方式，kinase cold-start, drug cold-start and kinase-drug cold-start, such that the model is tested on unseen proteins, unseen drugs or both. To use this option, set the argument --split_method using drug or both for the --split_method method, For example, to test on unseen drugs, run the following script.
```
python train_3dkkiba.py
```
### PDBBind v2020 and CASF-2016
**[1]** Feature extraction, encompassing the biochemical and conformational characteristics of kinase inhibitors, evolutionary information and the intricate spatial structural features of binding pockets and kinase domains.这里，为了更加快速的得到每个复合物体系的特征，我们首先建议您下载我们生成的所有的激酶，所有结合口袋以及所有激酶抑制剂的特征。然后执行以下命令：
```
python process_pdbbind2020.py
```
**[2]** 训练和测试模型在激酶-抑制剂结合亲和力的预测性能，我们提供了三种分割方式，kinase cold-start, drug cold-start and kinase-drug cold-start, such that the model is tested on unseen proteins, unseen drugs or both. To use this option, set the argument --split_method using drug or both for the --split_method method, For example, to test on unseen drugs, run the following script.
```
python train_pdbbind2020.py
```
## Other usages

### fine-tuning and predicion on G2019S LRRK2 mutant
```
python finetune_lrrk2g2019s.py
```

## Contact
Please submit GitHub issues or contact Yanan Tian(yanan.tian@mpu.edu.mo) for any questions related to the source code.
