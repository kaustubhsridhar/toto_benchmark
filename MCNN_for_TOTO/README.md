# Memory-Consistent Neural Networks
## Setup
Create env, install pytorch, install requirements:
```bash
conda create -n DL_env python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

Install this package:
```bash
pip install -e .
```

## Download assets, (MOCO) embeddings from creators iof the benchmark
Download:
```bash
python MCNN_for_TOTO/download_from_gdrive_and_extract.py
```

## Train MCNN
Create memories first:
```bash

```
