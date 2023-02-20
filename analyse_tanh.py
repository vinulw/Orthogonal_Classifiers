import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import os

if __name__=="__main__":
    data = pd.read_csv('tanh_jax_output.csv', sep=' *, *', engine='python')
    #print(data)
    columns = ['f0', 'decayRate', 'step', 'f', 'accuracy', 'cost']

    keys = data.keys()
    print(keys)
    pIndices = data['pIndex'].unique()
#    print(pIndices)

    parent_directory = os.getcwd()
    save_path = 'tanh_jax_figs/costs/'

    save_path = os.path.join(parent_directory, save_path)

    print(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for pind in tqdm(pIndices):
        datacurr = data[data['pIndex'] == pind]
        plt.figure()
        tstring = f'Decay Rate: {datacurr["decayRate"].iloc[0]}, f0: {datacurr["f0"].iloc[0]}, A: {datacurr["A"].iloc[0]}'
        plt.title(tstring)
        plt.plot(datacurr['step'], datacurr['cost'])
        plt.xlabel('step')
        plt.ylabel('cost')

        fname = f'{int(pind)}.png'
        fpath = os.path.join(save_path, fname)
        plt.savefig(fpath)
        plt.close()


