# BCMNet
## Requirements:
<ol>
<li>python 3.10 + torch 2.0.1 +torchvision 0.15.2</li>
<li>Install [Mamba](https://github.com/state-spaces/mamba) :`pip install causal-conv1d` and `pip install mamba-ssm` </li>
<li>Download code: `git clone https://github.com/Yiiii16/BCMNet.git`</li>
<li>`cd BCMNet/bcmnet` and run `pip install -e .`</li>
</ol>
## Preprocessing
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
## Train
```bash
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerbcmnet
```
## Inference
- Predict testing cases with `bcmnet` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f all -tr nnUNetTrainerbcmnet --disable_tta
```
## Acknowledgements
Thanks a lot to the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) , [Mamba](https://github.com/state-spaces/mamba) and [U-mamba](https://github.com/bowang-lab/U-Mamba) for making their code publicly available.
