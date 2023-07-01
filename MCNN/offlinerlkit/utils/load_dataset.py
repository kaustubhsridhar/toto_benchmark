import numpy as np
import torch
import collections
import pickle 

def qlearning_dataset_percentbc(task, chosen_percentage, num_memories_frac):
    folder = f'MCNN/memories'
    full_name = f'{task}_parsed_with_embeddings_moco_conv5'
    save_name = f'{folder}/{full_name}_updated_{num_memories_frac}_frac.pkl'
    with open(save_name, 'rb') as f:
            data = pickle.load(f)

    train_paths, memories_obs, memories_act, memories_next_obs, memories_rewards = data['train_paths'], data['memories_obs'], data['memories_act'], data['memories_next_obs'], data['memories_rewards']

    embeddings = np.concatenate([p['embeddings'] for p in train_paths])
    proprioriception = np.concatenate([p['observations'] for p in train_paths])
    observations = np.concatenate((proprioriception, embeddings), axis=1)
    actions = np.concatenate([path['actions'] for path in train_paths], axis=0)
    rewards = np.concatenate([path['rewards'] for path in train_paths], axis=0)
    next_embeddings = np.concatenate([p['next_embeddings'] for p in train_paths])
    next_proprioriception = np.concatenate([p['next_observations'] for p in train_paths])
    next_observations = np.concatenate((next_proprioriception, next_embeddings), axis=1)
    print(f'observations shape: {observations.shape}, actions shape: {actions.shape}, rewards shape: {rewards.shape}, next_observations shape: {next_observations.shape}')

    mem_observations = np.concatenate([path['mem_observations'] for path in train_paths], axis=0)
    mem_actions = np.concatenate([path['mem_actions'] for path in train_paths], axis=0)
    mem_next_observations = np.concatenate([path['mem_next_observations'] for path in train_paths], axis=0)
    mem_rewards = np.concatenate([path['mem_rewards'] for path in train_paths], axis=0)
    print(f'mem_observations shape: {mem_observations.shape}, mem_actions shape: {mem_actions.shape}, mem_rewards shape: {mem_rewards.shape}, mem_next_observations shape: {mem_next_observations.shape}')

    terminals = np.zeros_like(rewards)

    action_normalizer = np.max(np.abs(actions), axis=0, keepdims=True)
    actions = actions / action_normalizer
    mem_actions = mem_actions / action_normalizer

    return {
        'observations': observations,
        'actions': actions,
        'next_observations': next_observations,
        'rewards': rewards,
        'terminals': terminals,
        'mem_observations': mem_observations,
        'mem_actions': mem_actions,
        'mem_next_observations': mem_next_observations,
        'mem_rewards': mem_rewards,
        'memories_obs': memories_obs,
        'memories_actions': memories_act,
        'memories_next_obs': memories_next_obs,
        'memories_rewards': memories_rewards,
        'action_normalizer': action_normalizer,
    }

def qlearning_dataset(task, train_size, num_memories_frac):
    
    full_name = f'mems/updated_datasets/{task}_{train_size}_train/updated_{num_memories_frac}_frac.pkl'
    with open(full_name, 'rb') as f:
            data = pickle.load(f)

    train_paths, test_paths, memories_obs, memories_act, memories_next_obs, memories_rewards = data['train_paths'], data['test_paths'], data['memories_obs'], data['memories_act'], data['memories_next_obs'], data['memories_rewards']

    obs_dim = train_paths[0]['observations'].shape[1]
    act_dim = train_paths[0]['actions'].shape[1]

    observations = np.concatenate([path['observations'] for path in train_paths], axis=0)
    actions = np.concatenate([path['actions'] for path in train_paths], axis=0)
    next_observations = np.concatenate([path['next_observations'] for path in train_paths], axis=0)
    rewards = np.concatenate([path['rewards'] for path in train_paths], axis=0)
    
    for path in train_paths:
        path['terminals'][-1] = 1.0
    terminals = np.concatenate([path['terminals'] for path in train_paths], axis=0).astype(np.float32)

    mem_observations = np.concatenate([path['mem_observations'] for path in train_paths], axis=0)
    mem_actions = np.concatenate([path['mem_actions'] for path in train_paths], axis=0)
    mem_next_observations = np.concatenate([path['mem_next_observations'] for path in train_paths], axis=0)
    mem_rewards = np.concatenate([path['mem_rewards'] for path in train_paths], axis=0)

    test_observations = np.concatenate([path['observations'] for path in test_paths], axis=0)
    test_actions = np.concatenate([path['actions'] for path in test_paths], axis=0)
    test_next_observations = np.concatenate([path['next_observations'] for path in test_paths], axis=0)
    test_rewards = np.concatenate([path['rewards'] for path in test_paths], axis=0)
    
    for path in test_paths:
        path['terminals'][-1] = 1.0
    test_terminals = np.concatenate([path['terminals'] for path in test_paths], axis=0)
    
    test_mem_observations = np.concatenate([path['mem_observations'] for path in test_paths], axis=0)
    test_mem_actions = np.concatenate([path['mem_actions'] for path in test_paths], axis=0)
    test_mem_next_observations = np.concatenate([path['mem_next_observations'] for path in test_paths], axis=0)
    test_mem_rewards = np.concatenate([path['mem_rewards'] for path in test_paths], axis=0)

    return {
        'observations': observations,
        'actions': actions,
        'next_observations': next_observations,
        'rewards': rewards,
        'terminals': terminals,
        'mem_observations': mem_observations,
        'mem_actions': mem_actions,
        'mem_next_observations': mem_next_observations,
        'mem_rewards': mem_rewards,
        'memories_obs': memories_obs,
        'memories_actions': memories_act,
        'memories_next_obs': memories_next_obs,
        'memories_rewards': memories_rewards,
        'test_observations': test_observations,
        'test_actions': test_actions,
        'test_next_observations': test_next_observations,
        'test_rewards': test_rewards,
        'test_terminals': test_terminals,
        'test_mem_observations': test_mem_observations,
        'test_mem_actions': test_mem_actions,
        'test_mem_next_observations': test_mem_next_observations,
        'test_mem_rewards': test_mem_rewards,
    }


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, max_ep_len=1000, device="cpu"):
        super().__init__()

        self.obs_dim = dataset["observations"].shape[-1]
        self.action_dim = dataset["actions"].shape[-1]
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.device = torch.device(device)
        self.input_mean = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).mean(0)
        self.input_std = np.concatenate([dataset["observations"], dataset["actions"]], axis=1).std(0) + 1e-6

        data_ = collections.defaultdict(list)
        
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        self.trajs = []
        for i in range(dataset["rewards"].shape[0]):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000-1)
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                self.trajs.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1
        
        indices = []
        for traj_ind, traj in enumerate(self.trajs):
            end = len(traj["rewards"])
            for i in range(end):
                indices.append((traj_ind, i, i+self.max_len))

        self.indices = np.array(indices)
        

        returns = np.array([np.sum(t['rewards']) for t in self.trajs])
        num_samples = np.sum([t['rewards'].shape[0] for t in self.trajs])
        print(f'Number of samples collected: {num_samples}')
        print(f'Num trajectories: {len(self.trajs)}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_ind, start_ind, end_ind = self.indices[idx]
        traj = self.trajs[traj_ind].copy()
        obss = traj['observations'][start_ind:end_ind]
        actions = traj['actions'][start_ind:end_ind]
        next_obss = traj['next_observations'][start_ind:end_ind]
        rewards = traj['rewards'][start_ind:end_ind].reshape(-1, 1)
        delta_obss = next_obss - obss
    
        # padding
        tlen = obss.shape[0]
        inputs = np.concatenate([obss, actions], axis=1)
        inputs = (inputs - self.input_mean) / self.input_std
        inputs = np.concatenate([inputs, np.zeros((self.max_len - tlen, self.obs_dim+self.action_dim))], axis=0)
        targets = np.concatenate([delta_obss, rewards], axis=1)
        targets = np.concatenate([targets, np.zeros((self.max_len - tlen, self.obs_dim+1))], axis=0)
        masks = np.concatenate([np.ones(tlen), np.zeros(self.max_len - tlen)], axis=0)

        inputs = torch.from_numpy(inputs).to(dtype=torch.float32, device=self.device)
        targets = torch.from_numpy(targets).to(dtype=torch.float32, device=self.device)
        masks = torch.from_numpy(masks).to(dtype=torch.float32, device=self.device)

        return inputs, targets, masks