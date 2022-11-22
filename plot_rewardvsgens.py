import numpy as np
import matplotlib.pyplot as plt

run = 38 # select run to evaluate
save_plot = True
save_frames = False
savedir = ''

runstr = f'{run}'.zfill(3)
run_data = np.load(f'runs/run'+runstr+'/rewardsumss.npy')
individuals = run_data.shape[0]
generations = run_data.shape[1]

run_data_sort = np.sort(run_data, axis=0)

### PLOT RUN REWARDS ###
x = range(generations)
fig, ax = plt.subplots()
for indiv in range(individuals):
    plt.plot(x, np.sort(run_data_sort[indiv,:]))
ax.xaxis.get_major_locator().set_params(integer=True)
plt.xlim([0,generations-1])
plt.title(f'Evolution run {run}')
plt.xlabel('Generation')
plt.ylabel('Reward')
#plt.show()
if save_plot:
    plt.savefig(str(savedir+f'run{run}_plot.png'),bbox_inches='tight')

plt.clf()