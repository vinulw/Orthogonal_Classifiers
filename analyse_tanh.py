import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns

import os

def plot_steps(data=None):
    if data is None:
        data = pd.read_csv('tanh_output_100_long_run.csv', sep=' *, *', engine='python')
    #print(data)
    columns = ['f0', 'decayRate', 'step', 'f', 'accuracy', 'cost']

    keys = data.keys()
    print(keys)
    pIndices = data['pIndex'].unique()
    print(f'pIndices: {min(pIndices)} - {max(pIndices)}')

    parent_directory = os.getcwd()
    save_path = 'tanh_figs/accuracy/'

    save_path = os.path.join(parent_directory, save_path)

    print(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for pind in tqdm(pIndices):
        datacurr = data[data['pIndex'] == pind]
        plt.figure()
        tstring = f'Decay Rate: {datacurr["decayRate"].iloc[0]}, f0: {datacurr["f0"].iloc[0]}, A: {datacurr["A"].iloc[0]}'
        plt.title(tstring)
        plt.plot(datacurr['step'], datacurr['accuracy'])
        plt.xlabel('step')
        plt.ylabel('accuracy')

        pind = 100
        fname = f'{int(pind)}_long.png'
        fpath = os.path.join(save_path, fname)
        plt.savefig(fpath)
        plt.close()

if __name__=="__main__":
    data = pd.read_csv('tanh_jax_output.csv', sep=' *, *', engine='python')

    data_filt = data[data['step'] >= 90]
    grouped = data_filt.groupby(['pIndex']).mean().drop(columns=['f', 'step'])

    data0 = data[data['step'] == 0].filter(items=['pIndex', 'accuracy', 'cost'])
    data0 = data0.rename(columns={'accuracy': 'accuracy0', 'cost': 'cost0'})

    grouped = grouped.join(data0.set_index('pIndex'))
    grouped['Δ_accuracy'] = grouped['accuracy'] - grouped['accuracy0']
    grouped['Δ_cost'] = grouped['cost'] - grouped['cost0']
    grouped = grouped.reset_index()
    print(grouped)

    print(f'Accuracy range: {np.min(grouped["Δ_accuracy"]):.4f} - {np.max(grouped["Δ_accuracy"]):.4f}')
    print(f'Cost range: {np.min(grouped["Δ_cost"]):.4f} - {np.max(grouped["Δ_cost"]):.4f}')

    cost_range = [-0.15, 0.0]
    acc_range = [-0.5, 0.5]

    plt.plot(grouped['pIndex'], grouped['Δ_accuracy'], 'x')
    plt.xlabel('pIndex')
    plt.ylabel('Δ Accuracy')
    plt.savefig('pIndex_vs_daccuracy.png')
    plt.close()

    grouped = grouped.filter(items=['A', 'f0', 'decayRate', 'Δ_accuracy', 'Δ_cost'])

    group_A_f0 = grouped.drop(columns='decayRate').groupby(['A', 'f0']).mean()
    group_A_f0_acc = pd.pivot_table(group_A_f0, values='Δ_accuracy', index=['A'], columns=['f0'])
    group_A_f0_cost = pd.pivot_table(group_A_f0, values='Δ_cost', index=['A'], columns=['f0'])

    acc_cmap = sns.color_palette("coolwarm_r", as_cmap=True)


    xlabels = np.round(group_A_f0_acc.columns, 3)
    plt.figure()
    sns.heatmap(group_A_f0_acc, annot=True, fmt='.4f', xticklabels=xlabels, vmin=acc_range[0], vmax=acc_range[1], cmap=acc_cmap)
    plt.title('A vs f0 accuracy')
    plt.savefig('A_vs_f0_acc_heatmap.png')
    plt.close()

    xlabels = np.round(group_A_f0_cost.columns, 3)
    plt.figure()
    sns.heatmap(group_A_f0_cost, annot=True, fmt='.4f', xticklabels=xlabels, vmin=cost_range[0], vmax=cost_range[1])
    plt.title('A vs f0 cost')
    plt.savefig('A_vs_f0_cost_heatmap.png')
    plt.close()

    group_f0_decayRate  = grouped.drop(columns='A').groupby(['f0', 'decayRate']).mean()
    group_f0_decayRate_acc  = pd.pivot_table(group_f0_decayRate, values='Δ_accuracy', index=['f0'], columns=['decayRate'])
    group_f0_decayRate_cost  = pd.pivot_table(group_f0_decayRate, values='Δ_cost', index=['f0'], columns=['decayRate'])

    xlabels = np.round(group_f0_decayRate_acc.columns, 3)
    ylabels = np.round(group_f0_decayRate_acc.index, 3)
    plt.figure()
    sns.heatmap(group_f0_decayRate_acc, annot=True, fmt='.4f',
                xticklabels=xlabels, yticklabels=ylabels, vmin=acc_range[0],
                vmax=acc_range[1], cmap=acc_cmap)
    plt.title('f0 vs decayRate accuracy')
    plt.savefig('f0_vs_decayR_acc_heatmap.png')
    plt.close()

    xlabels = np.round(group_f0_decayRate_cost.columns, 3)
    ylabels = np.round(group_f0_decayRate_cost.index, 3)
    plt.figure()
    sns.heatmap(group_f0_decayRate_cost, annot=True, fmt='.4f',
                xticklabels=xlabels, yticklabels=ylabels, vmin=cost_range[0],
                vmax=cost_range[1])
    plt.title('f0 vs decayRate cost')
    plt.savefig('f0_vs_decayR_cost_heatmap.png')
    plt.close()

    group_A_decayRate  = grouped.drop(columns='f0').groupby(['A', 'decayRate']).mean()
    group_A_decayRate_acc  = pd.pivot_table(group_A_decayRate, values='Δ_accuracy', index=['A'], columns=['decayRate'])
    group_A_decayRate_cost  = pd.pivot_table(group_A_decayRate, values='Δ_cost', index=['A'], columns=['decayRate'])

    xlabels = np.round(group_A_decayRate_acc.columns, 3)
    plt.figure()
    sns.heatmap(group_A_decayRate_acc, annot=True, fmt='.4f', xticklabels=xlabels, vmin=acc_range[0], vmax=acc_range[1], cmap=acc_cmap)
    plt.title('A vs decayRate accuracy')
    plt.savefig('A_vs_decayRate_acc_heatmap.png')
    plt.close()

    xlabels = np.round(group_A_decayRate_cost.columns, 3)
    plt.figure()
    sns.heatmap(group_A_decayRate_cost, annot=True, fmt='.4f', xticklabels=xlabels, vmin=cost_range[0], vmax=cost_range[1])
    plt.title('A vs decayRate cost')
    plt.savefig('A_vs_decayRate_cost_heatmap.png')
    plt.close()

