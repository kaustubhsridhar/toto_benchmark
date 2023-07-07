# Memory-Consistent Neural Networks
Based on our submission to NeurIPS 2023 at [here](https://drive.google.com/file/d/1BkIQHdTJnlQq-Nnnd0400d1jBp7WIQIy/view).

## Setup
Create env, install pytorch, install requirements:
```bash
conda create -n MCNN_env python=3.8
conda activate MCNN_env
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -e .
pip install -r requirements.txt 
```

## Download (MOCO) embeddings from creators of the benchmark
Download:
```bash
python MCNN/download_from_gdrive.py
```

## Train MCNN
Create neural gas and updated dataset with memories:
```bash
mkdir MCNN/logs
nohup python -u MCNN/create_gng_incrementally.py --name pouring --num_memories_frac 0.1 > MCNN/logs/pouring_gng_0.1_frac.log &
nohup python -u MCNN/create_gng_incrementally.py --name scooping --num_memories_frac 0.1 > MCNN/logs/scooping_gng_0.1_frac.log &

CUDA_VISIBLE_DEVICES=0 nohup python -u MCNN/update_data.py --name pouring --num_memories_frac 0.1 > MCNN/logs/pouring_update_data_0.1_frac.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u MCNN/update_data.py --name scooping --num_memories_frac 0.1 > MCNN/logs/scooping_update_data_0.1_frac.log &
```

Train BC with memories:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u MCNN/td3bc_trainer.py --algo-name mem_bc --task pouring --num_memories_frac 0.1 --Lipz 1.0 --lamda 1.0 --use-tqdm 0 > MCNN/logs/pouring_mcnn_Lipz1.0_lamda1.0.log &
```
