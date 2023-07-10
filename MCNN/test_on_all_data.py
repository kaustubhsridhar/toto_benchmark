import numpy as np
import os
import pickle
import torch
from torchvision import transforms as T
from toto_benchmark.vision import load_model, load_transforms
import yaml 
from PIL import Image
from toto_benchmark.scripts.utils import Namespace

# load vision model, transforms
with open('toto_benchmark/outputs/collaborator_agent/hydra.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = Namespace(config)
vision_model = load_model(config)
transforms = load_transforms(config)

# load dataset with both raw images and downloaded embeddings
task = 'pouring'
num_memories_frac = 0.1
folder = f'MCNN/memories'
full_name = f'{task}_parsed_with_embeddings_moco_conv5'
save_name = f'{folder}/{full_name}_updated_{num_memories_frac}_frac.pkl'
with open(save_name, 'rb') as f:
    data = pickle.load(f)
with open(f'assets/{full_name}.pkl', 'rb') as f:
    orig_data = pickle.load(f)

# take a random image and its corresponding downloaded embedding
train_paths = orig_data

for path in train_paths:
    img_folder = path['traj_id']
    for pt_idx in range(len(path['observations'])):
        c_img_path = f"assets/cloud-data-{task}/data/{img_folder}/{path['cam0c'][pt_idx]}"
        d_img_path = f"assets/cloud-data-{task}/data/{img_folder}/{path['cam0d'][pt_idx]}"
        obs = path['observations'][pt_idx].reshape(-1)

        # actual embedding with vision_model
        c_img = Image.open(c_img_path)
        c_img_crop = c_img
        c_img_tensor = transforms(c_img_crop)
        c_embed = vision_model(torch.unsqueeze(c_img_tensor, dim=0)).detach().numpy().reshape(-1)

        # combine embedding with observation
        combined = np.concatenate((c_embed, obs), axis=0)

        # find closest memory

        
        # forward pass through MCNN


        # error with actual actions
        

