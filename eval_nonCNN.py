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

run = 1 # select run to evaluate
last_gen_best = True # select best indidual of last generation
save_plot, save_lastframe = False, False
save_frames = False
savedir = ''

runstr = f'{run}'.zfill(3)
run_data = np.load(f'runs/run'+runstr+'/rewardsumss.npy')
individuals = run_data.shape[0]
generations = run_data.shape[1]

### PLOT RUN REWARDS ###
x = range(generations)
fig, ax = plt.subplots()
for gen in range(individuals):
    ax.plot(x, np.sort(run_data[gen,:]))
ax.xaxis.get_major_locator().set_params(integer=True)
plt.xlim([0,generations-1])
plt.title(f'Evolution run {run}')
plt.xlabel('Generation')
plt.ylabel('Reward')
#plt.show()
if save_plot:
    plt.savefig(str(savedir+f'run{run}_plot.png'),bbox_inches='tight')
plt.clf()

### FIND BEST INDIVIDUAL ###
if last_gen_best:
    ind = (np.argmax(run_data[:,-1]),generations-1)
else:
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
p_prep = 2

### MODEL STRUCTURE ###
device = "cuda" if torch.cuda.is_available() else "cpu"
network_size = 64
model = net1(network_size).to(device)
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
env = gym_tetris.make('TetrisA-v3', deterministic = True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = False
state = env.reset()
first_step = True
rewardsum = 0
playfield_old = np.zeros([1,20,10], dtype=int)
perform_action = False
saveframe = 1000
frame_count = 0

while not done: # play until game over 
    # convert state (full-screen RBG matrix) to simplest form: 20x10 grid of 0/1
    playfield = state2playfield(state)
    playfield_prepped = playfield_prep(playfield,playfield_old,p_prep) # preprocess data for net, playing piece has values of 2
    
    # pass playfield to NN and determine action
    if perform_action:
        input = torch.from_numpy(playfield_prepped).float().to(device)
        with torch.inference_mode():
            logits = model(input)
        pred_probab = nn.Softmax(dim=1)(logits)
        pred = pred_probab.argmax(1)
        action = pred.detach().cpu().numpy()[0]
    else:
        action = 0
    perform_action = not perform_action

    # perform action
    state, reward, done, info = env.step(action)
    if save_frames and frame_count % saveframe == 0:
        np.save(f'best_run{run}_frame{frame_count}',state)
        print(f'frame {frame_count} saved')

    if first_step == True:
        stats_old = info['statistics']
        playfield_old = np.zeros([1,20,10], dtype=int)

    # calculate reward
    if info['statistics'] != stats_old: # if a new tetrimonio is introduced, evaluate reward
        height_rwd, edge_rwd = customreward(playfield, playfield_old, p_height_reward, p_edge_reward, False)
        rewardsum = rewardsum + height_rwd + np.sum(edge_rwd)
        stats_old = info['statistics']
        playfield_old = playfield
        
    if render:
        env.render()
    
    first_step = False
    frame_count += 1
    
env.close()
print('Reward: ', rewardsum)

#np.save('finalstate',state)

plt.imshow(state, interpolation='nearest')
plt.axis('off')
if save_lastframe:
    plt.savefig(str(savedir+f'run{run}_best.png'),bbox_inches='tight',pad_inches=0.0)

stoptimer = timeit.default_timer()
print('Time: ', round(stoptimer-starttimer,2))
