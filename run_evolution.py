### This file is used to perform one evolution run ###
import os
import timeit
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT, SIMPLE_MOVEMENT
import numpy as np
import random
import torch
from torch import nn
from funcs import *
starttimer = timeit.default_timer()
random.seed(0) # set random generators for repeatability
np.random.seed(0)

### EVOLUTION PARAMETERS ###
run = 1                 # ALWAYS SET A NEW RUN ID, OTHERWISE YOU OVERWRITE DATA
net_type = 0            # 0: non-CNN, 1: CNN
net_size = 64           # only for non-CNN
generations = 5
individuals = 10
indiv_best = 2          # number of best individuals used for mutation
determbool = True       # determinism of tetris game
prob_mut = 0.6          # mutation probability
p_mut = 0.4             # mutation factor
p_prep = 2              # playfield preparation factor for tertary representation
p_height_reward = 0     # reward factor for height reward
p_edge_reward = [7000,-2000] # reward factor for edgereward
p_rwd2 = 2              # reward double if piece tetromino falls perfectly
p_rwd_NOOP = 100        # reward factor for no-operation action

### CREATE SAVE DIR ###
path = 'runs/run' + f'{run}'.zfill(3)
if not os.path.exists(path):
    os.mkdir(path)
render = False # set to False for significant speedup

### MODEL STRUCTURE ###
device = "cuda" if torch.cuda.is_available() else "cpu"
if net_type == 0:
    model = net1(net_size).to(device)
else:
    model = net3().to(device)
size_dict, total_params = modelsizedict(model)

### INITIAL GENERATION TENSOR ###
ones = torch.ones([individuals,total_params]).float().to(device)
rand = torch.mul(torch.rand([individuals,total_params]).float().to(device),2)
gen_tensor = ones - rand
# gen_tensor = torch.load('runs/run020/gen_tensor_9.pt') # continue a run
rewardsums = np.zeros([individuals], dtype = int)
rewardsumss = np.zeros([individuals,generations], dtype=int)

offspring = individuals - indiv_best
offspring_ind = list(split(range(offspring), indiv_best))

### RUN EVOLUTION ###
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
        env = gym_tetris.make('TetrisA-v3', deterministic = determbool)
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
            if net_type == 0: # preprocess data for net, playing piece has values of p_prep
                playfield_prepped = playfield_prep(playfield,playfield_old,p_prep)
            else: 
                data_cut = state[48:208,96:176] # cutout of screen with RGB
                data_cut_T = np.transpose(np.expand_dims(data_cut, axis=0),(0,3,1,2)) # reorder shape for CNN 

            # pass playfield to NN and determine action
            if perform_action:
                if net_type == 0:
                    input = torch.from_numpy(playfield_prepped).float().to(device)
                else:
                    input = torch.from_numpy(data_cut_T.copy()).float().to(device)
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

stoptimer = timeit.default_timer()
print('Time: ', round(stoptimer-starttimer,2))
