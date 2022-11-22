import timeit
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT, SIMPLE_MOVEMENT
import numpy as np
import random
import torch
from torch import nn
from funcs import *
starttimer = timeit.default_timer()

run = 37 # select run to evaluate
save_fig = False
savedir = ''

runstr = f'{run}'.zfill(3)
run_data = np.load(f'runs/run'+runstr+'/rewardsumss.npy')
individuals = run_data.shape[0]
generations = run_data.shape[1]

p_rwd2 = 2 # reward double if piece tetromino falls perfectly
p_rwd_NOOP = 100 # reward factor for no-operation action

### PLOT RUN REWARDS ###
x = range(generations)
fig, ax = plt.subplots()
for gen in range(individuals):
    ax.plot(x, run_data[gen,:])
plt.xlim([0,generations-1])
plt.title(f'Evolution run {run}')
plt.xlabel('Generation')
plt.ylabel('Reward')
#plt.show()
if save_fig:
    plt.savefig(str(savedir+f'run{run}_plot.png'),bbox_inches='tight')
plt.clf()

ind = np.unravel_index(np.argmax(run_data, axis=None), run_data.shape)
print('Best Reward:', np.amax(run_data))
print(' Generation:', ind[1], '\n', 'Individual:', ind[0])

### SIMULATE BEST INDIVIDUAL ###
random.seed(0) # set random generators for repeatability
np.random.seed(0)
torch.manual_seed(0)

fname = str('runs/run'+runstr+f'/gen_tensor_{ind[1]}.pt')
best_ind = ind[0]

render = True # set to False for significant speedup
p_height_reward = 5    # reward factor for height reward
p_edge_reward = [3,-1] # reward factor for edgereward

### MODEL STRUCTURE ###
device = "cuda" if torch.cuda.is_available() else "cpu"
#network_size = 512
model = net3().to(device)
size_dict, total_params = modelsizedict(model)

### INITIAL GENERATION TENSOR ###
gen_tensor = torch.load(fname)

### SET MODEL PARAMETERS ###
idx = 0
for ((name, param), key) in zip(model.named_parameters(), size_dict):
    size = size_dict[key]
    start = idx
    end = idx + np.prod(size)
    param.data = torch.reshape(gen_tensor[best_ind,start:end].to(device),size)      
    idx = end

### PLAY TETRIS ###
env = gym_tetris.make('TetrisA-v3', deterministic = False)
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
    data_cut = state[48:208,96:176] # cutout of screen with RGB
    data_cut_T = np.transpose(np.expand_dims(data_cut, axis=0),(0,3,1,2)) # reorder shape for CNN
    
    # pass playfield to NN and determine action
    if perform_action:
        #input = torch.from_numpy(playfield_prepped).float().to(device)
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
        playfield_old = np.zeros([1,20,10], dtype=int)

    # calculate reward
    if info['statistics'] != stats_old: # if a new tetrimonio is introduced, evaluate reward
        height_rwd, edge_rwd = customreward(playfield, playfield_old, p_height_reward, p_edge_reward, False)
        rewardsum0 = rewardsum0 + height_rwd + np.sum(edge_rwd)
        stats_old = info['statistics']
        playfield_old = playfield
        
    if render:
        env.render()
    
    first_step = False
    
env.close()
print('Reward: ', rewardsum0)

#np.save('finalstate',state)

plt.imshow(state, interpolation='nearest')
plt.axis('off')
if save_fig:
    plt.savefig(str(savedir+f'run{run}_best.png'),bbox_inches='tight',pad_inches=0.0)

stoptimer = timeit.default_timer()
print('Time: ', round(stoptimer-starttimer,2))
