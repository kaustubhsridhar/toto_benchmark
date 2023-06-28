import numpy as np
import collections
import pickle
from MCNN.neural_gas_helpers import data2gas
import os 
from collections import defaultdict
import time
import argparse 
from tqdm import tqdm
from copy import deepcopy

# hyperparameters
parser = argparse.ArgumentParser(description='Parse constants.')
parser.add_argument('--name', default='pouring', type=str, help="")
parser.add_argument('--gng_epochs', default=1, type=int, help='num epochs for gng')
parser.add_argument('--num_memories_frac', type=float, default=0.1)
args = parser.parse_args()

# setup
folder = f'MCNN/memories'
os.makedirs(folder, exist_ok=True)

# load paths
full_name = f'{args.name}_parsed_with_embeddings_moco_conv5'
train_paths = pickle.load(open(f'assets/{full_name}.pkl', 'rb'))
num_train_points = np.sum([p['rewards'].shape[0] for p in train_paths])

# incremental neural gas creation
prev_graph = None

# num memories
num_memories = int(num_train_points * args.num_memories_frac)

# save name of gng file
gng_name = f'{folder}/{full_name}_memories_{args.num_memories_frac}_frac.pkl'
os.makedirs(f'{folder}', exist_ok=True)
t0 = time.time()

# create gng if it doesn't exist
if not os.path.exists(gng_name):
    # collect paths
    embeddings = np.concatenate([p['embeddings'] for p in train_paths])
    observations = np.concatenate([p['observations'] for p in train_paths])
    all_observations = np.concatenate((observations, embeddings), axis=1)
    all_actions = np.concatenate([p['actions'] for p in train_paths])
    print(f'{args.name} {all_observations.shape=}, {all_actions.shape=}')

    # create neural gas
    gng = data2gas(states=all_observations, max_memories=num_memories, gng_epochs=args.gng_epochs)

    with open(f'{gng_name}', 'wb') as f:
        pickle.dump(gng, f)
    print(f'==> for creating gng --- {full_name} {args.num_memories_frac=}, i.e. {num_memories=}/{num_train_points} duration={time.time()-t0}')
else:
    print(f'==> already exisiting gng --- {full_name} {args.num_memories_frac=}, i.e. {num_memories=}/{num_train_points} duration={time.time()-t0}')


