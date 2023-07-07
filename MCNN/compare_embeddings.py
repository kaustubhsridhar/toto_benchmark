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
path_idx = 1 # np.random.randint(len(train_paths))
pt_idx = 0 # np.random.randint(len(train_paths[path_idx]['observations']))

img_folder = train_paths[path_idx]['traj_id']
c_img_path = f"assets/cloud-data-{task}/data/{img_folder}/{train_paths[path_idx]['cam0c'][pt_idx]}"
d_img_path = f"assets/cloud-data-{task}/data/{img_folder}/{train_paths[path_idx]['cam0d'][pt_idx]}"
obs = train_paths[path_idx]['observations'][pt_idx]
downloaded_embed = train_paths[path_idx]['embeddings'][pt_idx]
print("Downloaded embedding shape:", downloaded_embed.shape)

# actual embedding with vision_model
c_img = Image.open(c_img_path)
c_img_crop = c_img
c_img_tensor = transforms(c_img_crop)

c_embed = vision_model(torch.unsqueeze(c_img_tensor, dim=0)).detach().numpy().reshape(-1)
print("color Embedding shape:", c_embed.shape)

# NOT A depth embedding model
# d_img = Image.open(d_img_path)
# d_img_crop = d_img
# d_img_tensor = transforms(d_img_crop)
# d_embed = vision_model(torch.unsqueeze(d_img_tensor, dim=0)).detach().numpy()
# print("color Embedding shape:", d_embed.shape)

# compare embeddings
diff = np.linalg.norm(c_embed - downloaded_embed)
print("Difference between embeddings:", np.abs(c_embed - downloaded_embed), "L2 norm", diff, "ratio", diff / np.linalg.norm(downloaded_embed))

"""
Outputs from above:
    Downloaded embedding shape: (2048,)
    color Embedding shape: (2048,)
    Difference between embeddings: [0.0000000e+00 4.7741923e-06 6.7055225e-06 ... 0.0000000e+00 2.5633723e-05
    0.0000000e+00] L2 norm 0.0033434865 ratio 0.0021146543
"""