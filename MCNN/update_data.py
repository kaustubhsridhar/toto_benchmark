import numpy as np
import collections
import pickle
from MCNN.neural_gas_helpers import data2gas
import os 
from collections import defaultdict
import time
import argparse 
import torch 
from tqdm import tqdm
from copy import deepcopy

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
parser = argparse.ArgumentParser(description='Parse constants.')
parser.add_argument('--name', default='puring', type=str, help="")
parser.add_argument('--num_memories_frac', type=float, default=0.05) 
args = parser.parse_args()
print(f'\n\n\n\n')

# setup
folder = f'MCNN/memories'
os.makedirs(folder, exist_ok=True)

# full_name, load paths
full_name = f'{args.name}_parsed_with_embeddings_moco_conv5'
train_paths = pickle.load(open(f'assets/{full_name}.pkl', 'rb'))
num_train_points = np.sum([p['rewards'].shape[0] for p in train_paths])

# save name of updated paths 
save_name = f'{folder}/{full_name}_updated_{args.num_memories_frac}_frac.pkl'
os.makedirs(f'{folder}', exist_ok=True)

if not os.path.exists(save_name):
    # load gng
    gng_name = f'{folder}/{full_name}_memories_{args.num_memories_frac}_frac.pkl'
    with open(f'{gng_name}', 'rb') as f:
        gng = pickle.load(f)
    node_weights = []
    for n in gng.graph.edges_per_node.keys():
        node_weights.append(n.weight)
    node_weights = np.array(node_weights)
    node_weights = torch.from_numpy(node_weights).float().to(device)

    # paths expanded
    embeddings = np.concatenate([p['embeddings'] for p in train_paths])
    observations = np.concatenate([p['observations'] for p in train_paths])
    all_obs = np.concatenate((observations, embeddings), axis=1)
    all_observations = deepcopy(all_obs[:-1])
    all_actions = np.concatenate([p['actions'][:-1] for p in train_paths])
    all_rewards = np.concatenate([p['rewards'][:-1] for p in train_paths])

    all_next_observations = deepcopy(all_obs[1:])

    all_observations_tensor = deepcopy(all_observations)
    all_observations_tensor = torch.from_numpy(all_observations_tensor).float().to(device)

    # find memories (aka points in train data closest to node_weights)
    t0 = time.time()
    print(f"Finding memories for {len(node_weights)} nodes' weights...")
    memories = []
    memories_actions = []
    memories_next_obs = []
    memories_rewards = []
    for w in node_weights:
        dists = torch.cdist(w, all_observations_tensor)
        min_dists, nearest_point = dists.min(dim=1)
        nearest_point = nearest_point.item()
        memories.append( all_observations[nearest_point] )
        memories_actions.append( all_actions[nearest_point] )
        memories_next_obs.append( all_next_observations[nearest_point] )
        memories_rewards.append( all_rewards[nearest_point] )
    memories = np.array(memories)
    memories_actions = np.array(memories_actions)
    memories_next_obs = np.array(memories_next_obs)
    memories_rewards = np.array(memories_rewards)
    print(f'{memories.shape=}, {memories_actions.shape=}, {memories_next_obs.shape=}, {memories_rewards.shape=}')
    memories = torch.from_numpy(memories).float().to(device)
    memories_actions = torch.from_numpy(memories_actions).float().to(device)
    memories_next_obs = torch.from_numpy(memories_next_obs).float().to(device)
    memories_rewards = torch.from_numpy(memories_rewards).float().to(device)
    print(f'Finding memories took {time.time() - t0} seconds')

    # update train paths
    t0 = time.time()
    print(f'Updating {len(train_paths)} train paths...')
    updated_train_paths = []
    for path in train_paths:
        new_path = deepcopy(path)
        obs = np.concatenate((path['observations'], path['embeddings']), axis=1)
        obs = torch.from_numpy(obs).float().to(device)
        
        # get nearest memories
        dists = torch.cdist(memories, obs)
        min_dists, nearest_memories = dists.min(dim=0)

        new_path['mem_observations'] = memories[nearest_memories].cpu().numpy()
        new_path['mem_actions'] = memories_actions[nearest_memories].cpu().numpy()
        new_path['mem_next_observations'] = memories_next_obs[nearest_memories].cpu().numpy()
        new_path['mem_rewards'] = memories_rewards[nearest_memories].cpu().numpy()

        updated_train_paths.append(new_path)
    print(f'Updating train paths took {time.time() - t0} seconds')

    # save
    data = {'train_paths': updated_train_paths, 
        'memories_obs': memories.cpu().numpy(), 
        'memories_act': memories_actions.cpu().numpy(), 
        'memories_next_obs': memories_next_obs.cpu().numpy(),
        'memories_rewards': memories_rewards.cpu().numpy()}
    with open(save_name, 'wb') as f:
        pickle.dump(data, f)
else:
    print(f'{save_name} already exists. Skipping.')

