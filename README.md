# SPU-BERT: Faster Human Multi-Trajectory Prediction from Socio-Physical Understanding of BERT.

<p align="center"><img src="overview.png" width="80%" height="80%"></p>

## Abstract
Pedestrian trajectory prediction should be based on social and physical understanding, considering movement patterns, nearby pedestrians, and surrounding obstacles simultaneously in a complex and crowded space. Also, it is necessary to generate multiple trajectories in the same situation to realize the multi-modality of human movement. In this paper, we propose SPU-BERT, a socially and physically acceptable multi-trajectory prediction for pedestrians using two sequential BERTs for multi-goal prediction (MGP) and trajectory-to-goal prediction (TGP) with fast computation. 
MGP consists of Transformer encoder and generative models to predict multiple goals. TGP with Transformer encoder generates multiple trajectories approaching the predicted goals of MGP. 
SPU-BERT can predict socio-physically acceptable multi-trajectory by understanding movements, social interactions, and scene contexts in trajectories and semantic map. In addition, the explainable results give confidence in the socio-physical understanding of SPU-BERT.



## Install
The codes are developed with python 3.6.
We tested the codes in python 3.6, 3.7, and 3.8.
Additional packages are included in the `requirements.txt`.
```bash
git clone https://github.com/kina4147/SPUBERT
export PYTHONPATH=$PYTHONPATH:$PWD
cd SPUBERT
pip install -r requirements.txt
```


### Dataset
Download datasets([data.zip](https://drive.google.com/file/d/1F80d4mEM9XXIJyaNhbBX9CSXDSDx-3Cy/view?usp=share_link)) including ETH/UCY dataset and Stanford drone dataset.
```bash
    ├── data                 # Datasets
    │    ├── ethucy          # ETH/UCY dataset
    │    └── sdd             # Stanford drone dataset
    ├── dataset              # files to load dataset: dataset.py, ethucy.py, sdd.py 
    ├── demo                 # executable files: train.py and test.py
    ├── output               # trained models
    │    ├── ethucy          # directory of the trained models for ETH/UCY datasets (eth, hotel, univ, zara1, zara2)
    │    └── sdd             # directory of the trained models for SDD (default)
    └── util
```

## Arguments

- `--cuda`, type=`bool`: The usage of GPU. 
- `--dataset_name`, type=`str`: The name of dataset among `ethucy` and `sdd`.
- `--dataset_split`, type=`str`: The split of dataset among `eth`, `hotel`, `univ`, `zara1`, and `zara2` for ETH/UCY dataset.
- `--output_name`, type=`str`: The name of output model 
- `--hidden`, type=`int`, default=`256`: The size of hidden state.
- `--layer`, type=`int`, default=`4`: The number of layers in Transformer encoders of MGP and TGP. 
- `--head`, type=`int`, default=`4`: The number of heads in Transformer encoders of MGP and TGP. 
- `--num_nbr`, type=`int`, default=`4`: The maximum number of neighbor pedestrians.
- `--view_angle`, type=`float`, default=`2.09`: The view angle to consider social interaction at the current frame.
- `--social_range`, type=`float`, default=`2`: The social range to consider social interaction at the current frame. 
- `--view_range`, type=`float`, default=`20`: The trajectory boundary range to filter out unnecessary trajectory positions.
- `--scene`, type=`bool`: The consideration of scene interaction 
- `--env_range`, type=`float`, default=`10`: The range of semantic map. 
- `--env_resol`, type=`float`, default=`0.2`: The resolution of semantic map.
- `--patch_size`, type=`int`, default=`16`: The size of patch for semantic map embedding.
- `--d_sample`, type=`int`, default=`1000`: The number of goal intention samples.

## Training
To train SPU-BERT, 
```bash
python -m demo.train  (--cuda) --dataset_name DATASET_NAME --dataset_split DATASET_SPLIT \
                       --output_name MODEL_NAME --d_sample NUM_GIS --num_nbr NUM_NEIGHBOR --scene 
```
For ETH/UCY dataset, `DATASET_SPLIT` can be `eth`, `hotel`, `univ`, `zara1`, `zara2`. 
For SDD, `DATASET_SPLIT` can be `default`. `--dataset_split` can be omitted because `default` is automatically set when `--dataset_name` is `sdd`.
The detailed description of arguments are explained in Argument below.
The trained model is save in `SPUBERT/output/DATASET_NAME/DATASET_SPLIT/MODEL_NAME.pth`.

## Evaluation
To test the trained SPU-BERT,
```bash
python -m demo.test (--cuda) --dataset_name DATASET_NAME --dataset_split DATASET_SPLIT \
                      --output_name MODEL_NAME --d_sample NUM_GIS --num_nbr NUM_NEIGHBOR --scene
```
All the arguments in the test should be the same with the arguments of the trained model (`MODEL_NAME.pth`) in the training.

## Pre-Trained Models
We uploaded the pretrained models described in the paper: SPU-BERT (H=256, L=4, A=4, D=1000), SPU-BERT<sub>*A*</sub> (H=256, L=4, A=8, D=5000), and SPU-BERT<sub>*HA*</sub> (H=512, L=4, A=8, D=5000).

|Models | ETH/UCY datasets <br/> (ADE<sub>20</sub> / FDE<sub>20</sub> (m)) | SDD <br/> (ADE<sub>20</sub> / FDE<sub>20</sub> (pixels)) | Computation Time (s)
|---- | :----: | :----: | :----: |
|SPU-BERT | 0.31/0.22  | 0.31/0.22 | 0.1 |
|SPU-BERT<sub>*A*</sub> | 0.31/0.22  | 0.31/0.22 | 0.1 |
|SPU-BERT<sub>*HA*</sub> | 0.31/0.22  | 0.31/0.22 | 0.1 |

Download the pretrained models ([output.zip](https://drive.google.com/file/d/1F80d4mEM9XXIJyaNhbBX9CSXDSDx-3Cy/view?usp=share_link)) and extract them as


```bash
    └── output               
         ├── ethucy 
         │    ├── eth ── spubert.pth / spubert_a.pth / spubert_ha.pth
         │    ├── hotel ── spubert.pth / spubert_a.pth / spubert_ha.pth
         │    ├── univ ── spubert.pth / spubert_a.pth / spubert_ha.pth
         │    ├── zara1 ── spubert.pth / spubert_a.pth / spubert_ha.pth
         │    └── zara2 ── spubert.pth / spubert_a.pth / spubert_ha.pth
         └── sdd   
              └── default ── spubert.pth / spubert_a.pth / spubert_ha.pth        
    
```  
To test SPU-BERT on the ETH-Hotel of ETH/UCY datasets, 
```bash
python -m demo.test (--cuda) --dataset_name ethucy --dataset_split hotel \
                     --hidden 256 --layer 4 --head 4 \
                     --d_sample 1000  --num_nbr 4 --scene --output_name spubert
```
To test SPU-BERT<sub>*A*</sub> on the UCY-Zara1 of ETH/UCY datasets, 
```bash
python -m demo.test (--cuda) --dataset_name ethucy --dataset_split zara1 \
                     --hidden 256 --layer 4 --head 8 \
                     --d_sample 5000  --num_nbr 4 --scene --output_name spubert_a
```
To test SPU-BERT<sub>*HA*</sub> on the SDD, 
```bash
python -m demo.test (--cuda) --dataset_name sdd \
                     --hidden 512 --layer 4 --head 8 \
                     --d_sample 5000  --num_nbr 4 --scene --output_name spubert_ha
```


## Thanks
- Transformers used in this model comes from [HugginFace](https://huggingface.co/).
- ETH/UCY Dataset and SDD come from [Y-Net](https://github.com/HarshayuGirase/Human-Path-Prediction).  


## Citation  
If you find this work useful, it would be grateful to cite our paper!

```bib
@article{na2022knowledge,
  title={SPU-BERT: Faster Human Multi-Trajectory Prediction from Socio-Physical Understanding of BERT},
  author={Ki-In Na, Ue-Hwan Kim, and Jong-Hwan Kim},
  journal={Knowledge Based Systems (submitted)},
  year={2022}
}
```

## Contact us

Ki-In Na ([@kina4147](https://github.com/kina4147)): kina4147@gmail.com  
