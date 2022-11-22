import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT, SIMPLE_MOVEMENT

class net1(nn.Module):
    def __init__(self,size):
        super(net1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20*10, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 5),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class net2(nn.Module):
    def __init__(self,size):
        super(net2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20*10, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 5),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class net3(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 17 * 37, 120) # 17*37 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Conv1 -> (2,2) max pooling
        x = self.pool(F.relu(self.conv2(x))) # Conv2 -> (2,2) max pooling
        x = torch.flatten(x, 1) # flatten all dimensions
        x = F.relu(self.fc1(x)) # fully connected 1, relu activation
        x = F.relu(self.fc2(x)) # fully connected 2, relu activation 
        x = self.fc3(x) # output layer
        return x

class net4(nn.Module):
    def __init__(self,size):
        super(net4, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20*10, size),
            nn.Tanh(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 5),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def modelsizedict(nnmodel): # returns dictionary of model layers and their size, last entry is total
    names = []
    sizes = []
    for name, param in nnmodel.named_parameters():
        #if param.requires_grad:
        names.append(name)
        sizes.append([*param.data.shape])
    
    total_params = 0
    size_dict = {key: value for key, value in zip(names, sizes)}
    for i in size_dict.values():
        mult = np.prod(i)
        total_params += mult

    return size_dict, total_params

def state2playfield(state): # convert state (rbg pixel output) to simple 10x20 matrix
    topleft = [48,96] # coordinate of top left pixel [COLUMN ROW]
    playfield = np.zeros([1,20,10], dtype=int)
    for i in range(20):
        for j in range(10):
            if state[topleft[0]+(i*8),topleft[1]+(j*8),1] != 0:
                playfield[0,i,j] = 1
            else:
                playfield[0,i,j] = 0
    
    return playfield

def playfield_prep(pf_new: np.ndarray,pf_old: np.ndarray,p_prep): # convert state (rbg pixel output) to simple 10x20 matrix
    playfield = np.zeros([1,20,10], dtype=int)
    placement_ind = np.nonzero(pf_new - pf_old) # indices of last placed tetrimonio
    playfield[placement_ind] = p_prep
    playfield += pf_old

    return playfield

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def customreward(pf_new: np.ndarray, pf_old: np.ndarray, p_height_reward: float, p_edge_reward: list, count_diagonal: bool):
    pf_old_pad = np.array([np.pad(pf_old, 1, pad_with, padder=1)[1]]) # pad border of playfield with ones
    pf_new_pad = np.array([np.pad(pf_new, 1, pad_with, padder=1)[1]])
    pf_old_pad[:,0,1:-1] = 0 # make top row zeros again
    pf_new_pad[:,0,1:-1] = 0

    placement_ind = np.nonzero(pf_new_pad - pf_old_pad) # indices of last placed tetrimonio
    
    adj_ind = []
    for X,Y in zip(placement_ind[1],placement_ind[2]): # create list of all indices adjacent to placement_ind in X,Y and diagonally
        adj_ind.append([X+1,Y])
        adj_ind.append([X,Y+1])
        adj_ind.append([X-1,Y])
        adj_ind.append([X,Y-1])

        if count_diagonal:
            adj_ind.append([X+1,Y+1])        
            adj_ind.append([X-1,Y+1])        
            adj_ind.append([X-1,Y-1])        
            adj_ind.append([X+1,Y-1])

    adj_ind = np.unique(adj_ind, axis=0) # remove duplicates
    adj_reformat_ind = (np.zeros([len(adj_ind)],dtype=int),adj_ind[:,0],adj_ind[:,1]) # reformat to separate arrays for each dimension

    pf_adj = np.zeros([1,22,12], dtype=int)

    pf_adj[adj_reformat_ind] = 2 # create array with the adjacent indices

    overlap = pf_old_pad + pf_adj
    
    edge_reward = [np.count_nonzero(overlap == 3)*p_edge_reward[0], np.count_nonzero(overlap == 2)*p_edge_reward[1]] # count overlapping pixel
    height_reward = np.amin(placement_ind[1])*p_height_reward # reward for low placement of tetromino

    return height_reward, edge_reward

def customreward2(pf_new: np.ndarray, pf_old: np.ndarray, p_rwd2: float):
    pf_active = pf_new - pf_old
    
    if not pf_active.any(): # if no active tetromino is present, skip reward calculation
        reward = 0
        return reward
    
    placement_ind = np.nonzero(pf_active)
    column_ind = np.unique(placement_ind[2], axis=0) # active tetromino columns
    distance_vec = np.zeros_like(column_ind)
    
    if np.max(placement_ind[1]) < 19: # if tetromino is not at bottom, calculate reward
        for i in range(column_ind.size):
            top = np.max(np.nonzero(pf_active[:,:,column_ind[i]])[1])
            
            if np.sum(pf_old[:,:,column_ind[i]]) == 0: # if no tetrominos underneath
                distance_vec[i] = 20 - top
            
            else:
                bot = np.min(np.nonzero(pf_old[:,:,column_ind[i]])[1])
                distance_vec[i] = bot - top

        reward = int(np.average(distance_vec))
        if np.all(distance_vec == distance_vec[0]): # if all distances are the same (perfect fit)
            reward = reward * p_rwd2 # multiply reward

    else:
        reward = 0 

    return reward

def play_tetris(genome):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network_size = 64
    model = net1(network_size).to(device)
    size_dict, total_params = modelsizedict(model)

    ### SET MODEL PARAMETERS ###
    idx = 0
    for ((_, param), key) in zip(model.named_parameters(), size_dict):
        size = size_dict[key]
        start = idx
        end = idx + np.prod(size)
        param.data = torch.reshape(genome[start:end].to(device),size)      
        idx = end

    ### PLAY TETRIS ###
    env = gym_tetris.make('TetrisA-v3', deterministic = True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    done = False
    state = env.reset()
    first_step = True
    rewardsum = 0

    while not done: # play until game over
        # convert state (full-screen RBG matrix) to simplest form: 20x10 grid of 0/1
        playfield = state2playfield(state)
        
        # pass playfield to NN and determine action
        input = torch.from_numpy(playfield).float().to(device)
        with torch.inference_mode():
            logits = model(input)
        pred_probab = nn.Softmax(dim=1)(logits)
        pred = pred_probab.argmax(1)
        action = pred.detach().cpu().numpy()[0]

        # perform action
        state, reward, done, info = env.step(action)

        if first_step == True:
            stats_old = info['statistics']
            playfield_old = np.zeros([1,20,10], dtype=int)

        # calculate reward
        if info['statistics'] != stats_old: # if a new tetrimonio is introduced, evaluate reward
            _ , reward1 = customreward(playfield,playfield_old,1,[3,-2],False)
            rewardsum = rewardsum + np.sum(reward1)
            stats_old = info['statistics']
            playfield_old = playfield
            
        # if render:
        #     env.render()
        
        first_step = False
        
    env.close()
    print(f'Reward: {rewardsum}')
    return rewardsum

def run_evolution(settings: list):
    """
    SETTINGS FORMAT:
    0:  run #   int
    1:  gens    int
    2:  indv    int
    3:  best    int
    4:  prob_m  float
    5:  p_mut   float
    6:  p_prep  int
    7:  p_h     int
    8:  p_e     list
    9:  p_rwd2  int
    10: p_NOOP  int
    """
    ### CREATE SAVE DIR ###
    run = settings[0]
    path = 'runs/run' + f'{run}'.zfill(3)
    if not os.path.exists(path):
        os.mkdir(path)

    ### EVOLUTION PARAMETERS ###
    generations = settings[1]
    individuals = settings[2]
    indiv_best = settings[3] # number of best individuals used for mutation
    prob_mut = settings[4] # mutation probability
    p_mut = settings[5]  # mutation factor
    p_prep = settings[6] # playfield prep factor for tertary representation
    render = False # set to False for significant speedup
    offspring = individuals - indiv_best
    offspring_ind = list(split(range(offspring), indiv_best))
    p_height_reward = settings[7]    # reward factor for height reward
    p_edge_reward = settings[8] # reward factor for edgereward
    p_rwd2 = settings[9] # reward double if piece tetromino falls perfectly
    p_rwd_NOOP = settings[10] # reward factor for no-operation action

    ### MODEL STRUCTURE ###
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network_size = 64
    model = net1(network_size).to(device)
    size_dict, total_params = modelsizedict(model)

    ### INITIAL GENERATION TENSOR ###
    ones = torch.ones([individuals,total_params]).float().to(device)
    rand = torch.mul(torch.rand([individuals,total_params]).float().to(device),2)
    gen_tensor = ones - rand
    # gen_tensor = torch.load('runs/run020/gen_tensor_9.pt') # continue run 20

    rewardsums = np.zeros([individuals], dtype = int)
    rewardsumss = np.zeros([individuals,generations], dtype=int)

    for gen in range(generations):
        print('GENERATION:    ', gen)
        torch.save(gen_tensor, str(path+'/gen_tensor_'+str(gen)+'.pt'))

        ### APPLY MUTATION ###
        if gen != 0:
            ind = np.argpartition(rewardsums, -indiv_best)[-indiv_best:] # find indices with best rewrard
            print('BEST:  ', ind)
            gen_tensor[0:indiv_best,:] = gen_tensor[ind,:]
            j = 0
            for section in offspring_ind:
                for i in section:
                    gene_idx = torch.randperm(gen_tensor.size(1))[:int(total_params*prob_mut)].to(device) # Randomly select percentage of genes to mutate
                    mutation = torch.mul(torch.ones(gene_idx.size(0)).float().to(device) - torch.mul(torch.rand(gene_idx.size(0)).float().to(device),2),p_mut)
                    gen_tensor[i+indiv_best,gene_idx] = gen_tensor[j,gene_idx] + mutation           
                j += 1

        ### EVALUATE GENERATION ###
        for row in range(individuals):
            print('INDIVIDUAL: ', row)
            ### SET MODEL PARAMETERS ###
            idx = 0
            for ((name, param), key) in zip(model.named_parameters(), size_dict):
                size = size_dict[key]
                start = idx
                end = idx + np.prod(size)
                param.data = torch.reshape(gen_tensor[row,start:end].to(device),size)      
                idx = end

            ### PLAY TETRIS ###
            env = gym_tetris.make('TetrisA-v3', deterministic = True) # CHANGE
            env = JoypadSpace(env, SIMPLE_MOVEMENT)

            done = False
            state = env.reset()
            first_step = True
            playfield_old = np.zeros([1,20,10], dtype=int)
            perform_action = False
            rewardsum0 = 0
            rewardsum1 = 0
            rewardsum2 = 0
        
            while not done: # play until game over
                # convert state (full-screen RBG matrix) to simplest form: 20x10 grid of 0/1
                playfield = state2playfield(state)
                playfield_prepped = playfield_prep(playfield,playfield_old,p_prep) # preprocess data for net, playing piece has values of p_prep
                
                # pass playfield to NN and determine action
                if perform_action:
                    input = torch.from_numpy(playfield_prepped).float().to(device)
                    with torch.inference_mode():
                        logits = model(input)
                    pred_probab = nn.Softmax(dim=1)(logits)
                    pred = pred_probab.argmax(1)
                    action = pred.detach().cpu().numpy()[0]
                    if action == 0:
                        rewardsum2 += p_rwd_NOOP
                else:
                    action = 0
                
                perform_action = not perform_action
                # print(perform_action)
                # print('action:  ', action)

                # perform action
                state, reward, done, info = env.step(action)

                if first_step == True:
                    stats_old = info['statistics']

                # calculate rewards
                rewardsum1 += customreward2(playfield,playfield_old,p_rwd2)
                
                if info['statistics'] != stats_old: # if a new tetromino is introduced, evaluate reward
                    height_rwd, edge_rwd = customreward(playfield,playfield_old,p_height_reward,p_edge_reward,False)
                    rewardsum0 = rewardsum0 + height_rwd + np.sum(edge_rwd)
                    #print(height_rwd,edge_rwd)
                    # define current stats and playfield old
                    stats_old = info['statistics']
                    playfield_old = playfield
                    
                if render:
                    env.render()
                
                first_step = False
                
            env.close()
            rewardsums[row] = rewardsum0 + rewardsum1 + rewardsum2
            print('Reward: ', rewardsums[row])

        rewardsumss[:,gen] = rewardsums
        print(rewardsums, '\n')

    print(rewardsumss)
    np.save(path+'/rewardsumss.npy',rewardsumss)