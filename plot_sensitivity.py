import numpy as np
import matplotlib.pyplot as plt

runs = np.array([39,40,41,42]) # select runs to evaluate in single plot
save_plot = False
savedir = ''

runstr = f'{runs[0]}'.zfill(3)
run0_data = np.load(f'runs/run'+runstr+'/rewardsumss.npy')
individuals = run0_data.shape[0]
generations = run0_data.shape[1]

final_rwds = []
mean_SD = np.zeros((runs.size*3,generations))

idx = 0
for run in runs:
    runstr = f'{run}'.zfill(3)
    data = np.load(f'runs/run'+runstr+'/rewardsumss.npy')
    final_rwds.append(data[:,-1])

    stds = np.std(data, axis=0, ddof=1)
    means = np.mean(data, axis=0)
    upper,lower = means + stds, means - stds
    mean_SD[idx*3,:] = means
    mean_SD[idx*3+1,:] = lower
    mean_SD[idx*3+2,:] = upper
    
    idx += 1

# final_rwds0 = [np.append(final_rwds[0],final_rwds[1]),np.append(final_rwds[2],final_rwds[3])]

### VIOLIN PLOT ###
fig, ax = plt.subplots()
ax.violinplot(final_rwds,showmeans=False,showmedians=True)
ax.yaxis.grid(True)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax.set_xticks([y + 1 for y in range(len(final_rwds))],labels=['80%', '60%', '40%', '20%'])
plt.title(f'Final reward for diffent probabilities of mutation')
plt.xlabel('probability of mutation')
plt.ylabel('Reward')
plt.show()
if save_plot:
    plt.savefig(str(savedir+f'p_prob_violin2.pdf'), format='pdf', bbox_inches='tight')
plt.clf()


### PLOT RUN REWARDS ###
# run_data_sort = np.sort(run_data, axis=0)
# x = range(generations)
# fig, ax = plt.subplots()
# for indiv in range(individuals):
#     plt.plot(x, np.sort(run_data_sort[indiv,:]))
# ax.xaxis.get_major_locator().set_params(integer=True)
# plt.xlim([0,generations-1])
# plt.title(f'Evolution run {run}')
# plt.xlabel('Generation')
# plt.ylabel('Reward')
# #plt.show()
# if save_plot:
#     plt.savefig(str(savedir+f'run{run}_plot.png'),bbox_inches='tight')
# plt.clf()

### STATISTICAL PLOTS ###
# fig, ax = plt.subplots()
# ax.fill_between(x, lower, upper, facecolor='red', alpha=0.2)
# ax.plot(x, means, color='red')
# ax.xaxis.get_major_locator().set_params(integer=True)
# plt.xlim([0,generations-1])
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.title(f'Evolution run {run}')
# plt.xlabel('Generation')
# plt.ylabel('Reward')
# plt.show()
# if save_plot:
#     plt.savefig(str(savedir+f'run{run}_plot.png'),bbox_inches='tight')
# plt.clf()

# fig, ax = plt.subplots()
# ax.plot(x, means, color='red')
# ax.plot(x, lower, x, upper, 'r--', alpha=0.3)
# ax.xaxis.get_major_locator().set_params(integer=True)
# plt.xlim([0,generations-1])
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.title(f'Evolution runs {runs[0]}-{runs[-1]}')
# plt.xlabel('Generation')
# plt.ylabel('Reward')
# plt.show()
# if save_plot:
#     plt.savefig(str(savedir+f'run{run}_plot.png'),bbox_inches='tight')
# plt.clf()

### REWARD MEAN SD PLOT ###
# cases = [50,25,10,5,2]
# fig, ax = plt.subplots()
# for i in range(runs.size):
#     color = next(ax._get_lines.prop_cycler)['color']
#     ax.plot(x, mean_SD[i*3], color=color, label=str(cases[i]))
#     ax.plot(x, mean_SD[i*3+1], linestyle='--', color=color)
#     ax.plot(x, mean_SD[i*3+2], linestyle='--', color=color)
#     lgd = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
# ax.get_legend().set_title('# best\nindividuals')
# ax.xaxis.get_major_locator().set_params(integer=True)
# plt.xlim([0,generations-1])
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.title(f'Reward mean with SD over generations')
# plt.xlabel('Generation')
# plt.ylabel('Reward')
# #plt.show()
# if save_plot:
#     plt.savefig(str(savedir+f'ind_best_sens.pdf'), format='pdf', bbox_inches='tight')
# plt.clf()

### BOX PLOT ###
# fig, ax = plt.subplots()
# ax.boxplot(final_rwds)
# ax.yaxis.grid(True)
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax.set_xticks([y + 1 for y in range(len(final_rwds))],labels=['50', '25', '10', '5', '2'])
# plt.title(f'Boxplot of final reward values')
# plt.xlabel('# best individuals')
# plt.ylabel('Reward')
# plt.show()
# if save_plot:
#     plt.savefig(str(savedir+f'ind_best_box.pdf'), format='pdf', bbox_inches='tight')
# plt.clf()

