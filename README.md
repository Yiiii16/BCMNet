# BCMNet
Repository for the paper “A Dual-Branch Network for Simultaneous Liverand Tumor Segmentation Based on Mamba andBoundary-Aware Module”
## Requirements:
1.python 3.10 + torch 2.0.1 +torchvision 0.15.2

2.Install [Mamba](https://github.com/state-spaces/mamba) :`pip install causal-conv1d` and `pip install mamba-ssm` 

3.Download code: `git clone https://github.com/Yiiii16/BCMNet.git` 

4.`cd BCMNet/bcmnet` and run `pip install -e .`


## Preprocessing
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
## Train
```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerbcmnet
```
## Inference
- Predict testing cases with `bcmnet` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f all -tr nnUNetTrainerbcmnet --disable_tta
```
