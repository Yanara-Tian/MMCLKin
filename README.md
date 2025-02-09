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
This package is tested with Python 3.8.15 and CUDA 11.0 on Ubuntu 20.04. Run the following to create a conda environment and install the required Python packages (modify `torch+cu116` according to your CUDA version). 
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
pip install jupyterlab
pip install notebook
```
Running the above lines of `pip install` should be sufficient to install all  MMCLKin's required packages (and their dependencies). Specific versions of the packages we tested were listed in `requirements.txt`.

## Reproduce Results

### Kinase-inhibitor binding affinity 

#### Predictive performance of kinase-inhibitor binding affinity on the 3DKDavis dataset
**[1]** Download checkpoints(~298MB) and dataset(~15GB) to main directory, and then extract them.
```
https://fca_icdb.mpu.edu.mo/DataResource/database/pkls/3dkdavis_new_kinase_affinity.tar.gz
tar zxvf 3dkdavis_new_kinase_affinity.tar.gz -C ./pkls
https://fca_icdb.mpu.edu.mo/DataResource/database/test_datasets/3dkdavis_new_kinase_affinity.tar.gz
tar zxvf 3dkdavis_new_kinase_affinity.tar.gz -C ./test_datasets
```
**[2]** Test the prediction performance of MMCLKin for kinase-inhibitor binding affinity on 3DKDavis dataset. We offer dataset splits based on drug cold-start, kinase cold-start, or kinase-drug cold-start, ensuring that the model is tested on unseen kinases, unseen drugs, or both. To enable this option, set the --label argument to new_kinase, new_drug, or both_new. For example, to test on unseen kinases, run the following script:
```
python test_3dkdavis_affinity.py
```
#### Predictive performance of kinase-inhibitor binding affinity on the low sequence similarity dataset of 3DKKIBA
**[1]** Download checkpoints(~300MB) and dataset(~6.8GB)to main directory, and then extract the dataset.
```
https://fca_icdb.mpu.edu.mo/DataResource/database/pkls/3dkkiba_new_kinase_affinity.tar.gz
tar zxvf 3dkkiba_new_kinase_affinity.tar.gz -C ./pkls
https://fca_icdb.mpu.edu.mo/DataResource/database/test_datasets/3dkkiba_new_kinase_affinity.tar.gz
tar zxvf 3dkkiba_new_kinase_affinity.tar.gz -C ./test_datasets
```
**[2]** Test the prediction performance of MMCLKin for kinase-inhibitor binding affinity on low sequence similarity dataset of 3DKDavis. For example, to evaluate on unseen kinases, run the following script:
```
python test_3dkkiba_affinity.py
```

### The selectivity of kinase inhibitors on human kinome

#### Predictive performance of the selectivity of kinase inhibitors on the 3DKDavis dataset
**[1]** Download checkpoints(~300MB) and dataset(~15GB) to main directory, and then extract the dataset.
```
https://fca_icdb.mpu.edu.mo/DataResource/database/pkls/3dkdavis_selectivity.tar.gz
tar zxvf 3dkdavis_selectivity.tar.gz -C ./pkls
https://fca_icdb.mpu.edu.mo/DataResource/database/test_datasets/3dkdavis_new_drug_selectivity.tar.gz
tar zxvf 3dkdavis_new_drug_selectivity.tar.gz -C ./test_datasets
```
**[2]** To ensure comprehensive learning of human kinases, the predictive performance for kinase inhibitor selectivity of MMCLKin was evalueated under the drug cold-start splitting strategy.
```
python test_3dkdavis_selectivity.py
```
#### Predictive performance of the selectivity of kinase inhibitor on the low sequence similarity dataset of 3DKKIBA
**[1]** Download checkpoints(~300MB) and dataset(~6.1GB) to main directory, and then extract the dataset.
```
https://fca_icdb.mpu.edu.mo/DataResource/database/pkls/3dkkiba_selectivity.tar.gz
tar zxvf 3dkkiba_selectivity.tar.gz -C ./pkls
https://fca_icdb.mpu.edu.mo/DataResource/database/test_datasets/3dkkiba_new_drug_selectivity.tar.gz
tar zxvf 3dkkiba_new_drug_selectivity.tar.gz -C ./test_datasets
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
**[1]** Download the new constructed 3DKDavis dataset and extract its content.
```
wget 3dkdavis/3dkdavis.tar.gz
tar zxvf 3dkdavis.tar.gz
```
**[2]** Feature extraction, encompassing the biochemical and conformational characteristics of kinase inhibitors, evolutionary information and the intricate spatial structural features of binding pockets and kinase domains.To expedite feature generation for each complex, we recommend executing the following command, which directly generates complex-level features using our pre-generated feature files for all kinases, binding pockets, and kinase inhibitors:
```
python process_3dkdavis.py
```
**[3]** Three splitting strategies (kinase cold-start, drug cold-start and kinase-drug cold-start) remain available for training and evaluating the predictive performance of MMCLKin on kinase-inhibitor binding affinity. For example, to train and test on unseen kinases, run the following script:
```
python train_3dkdavis.py
```
For the selectivity of kinase inhibitors, only the drug cold-start splitting method is provided，execute the following command:
```
python train_3dkdavis_selectivity.py
```
### 3DKKIBA
**[1]** Download the new constructed low sequence similarity dataset of 3DKDavis and extract its content.
```
wget 3dkkiba/30sm_kiba_gra_seq.tar.gz
tar zxvf 30sm_kiba_gra_seq.tar.gz
```
If you wish to obtain features for the entire 3DKKIBA dataset (approximately 500GB), download the 3dkkiba.tar.gz and then execute the following command:
```
wget 3dkkiba/3dkkiba.tar.gz
tar zxvf 3dkkiba.tar.gz
python process_3dkkiba.py
```
**[2]** Training and testing MMCLKin on the dataset with low protein sequence similarity. For kinase-inhibitor binding affinity, three splitting strategies are provided. For example, to test on unseen kinases, run the following script.
```
python train_3dkkiba.py
```
For the selectivity of kinase inhibitors on datasets with low protein sequence similarity，execute the following command:
```
python train_3dkkiba_selectivity.py
```
## Other usages
### PDBBind v2020 and CASF-2016
**[1]** Download the PDBBindv2020 dataset and extract its content.
```
wget pdbbind2020/pdbbind2020.tar.gz
tar zxvf pdbbind2020.tar.gz
```
**[2]** extracting the features of PDBBindv2020 and CASF-2016 datasets.
```
python process_pdbbind2020.py
```
**[3]** Training and testing the generalization ability of MMCLKin for protein-drug binding affinity on the dataset with diversity protein structures.
```
python train_pdbbind2020.py
```

### Fine-tuning and predicion on G2019S LRRK2 mutant
**[1]** Download checkpoints(~300MB) and a dataset composed of a 1:1 ratio of inhibitors targeting both wild-type LRRK2 and LRRK2 G2019S  mutant kinases.
```
wget pkls/finetune_lrrk2g2019s/MMCLKin_DTI_pearson_best.pkl
wget mutant/lrrk2_g4.tar.gz
tar zxvf lrrk2_g4.tar.gz
wget mutant/lrrk2_mw.tar.gz
tar zxvf lrrk2_mw.tar.gz
```
**[2]** Fine-tune the MMCLKin model, and then perform prediction and validation on the inhibitors identified by our group. Execute the following command:
```
python finetune_lrrk2g2019s.py
```
## Contact
Please submit GitHub issues or contact Yanan Tian(yanan.tian@mpu.edu.mo) for any questions related to the source code.
