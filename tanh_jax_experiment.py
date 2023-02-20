'''
Experiment to tune hyperparameters for tanh cost function training.

We have the following hyperparameters to train:
    - f0 : [0.02, 0.2]
    - decayRate: [0.05, 0.5]
    - A: [10, 1000]

Will have 5 steps for each of the parameters. Leading to 125 experiments in total

This code uses the jax implementation of the ZOverlaps and dOdV
'''
import numpy as np
import jax.numpy as jnp
from jax.config import config
from tqdm import tqdm
from stacking_simple import evaluate_classifier_top_k_accuracy, load_data
from stacking_simple import labelsToBitstrings, apply_U
from stacking_jax import calculate_tanhCost
from stacking_jax import update_U_jax as update_U
from stacking_jax import generate_Zi


def save_output(fname, data):
    with open(fname, 'a') as f:
        np.savetxt(f, data, delimiter=',')


config.update("jax_enable_x64", True)

parameters = np.loadtxt('hyperparameters.csv', delimiter=',')

np.random.seed(1)
prefix = "data_dropbox/mnist/"
trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

N = 1000
Nsteps = 100
outputCost = True
Nq = 4

Zis = []

U = np.eye(16)

for i in range(4):
    Zis.append(generate_Zi(Nq, i+1))
Zis = [jnp.array(Zi) for Zi in Zis]

trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
          prefix + trainingLabelPath,
          N)

trainingPredJ = jnp.array(trainingPred)

acc = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)

U = np.eye(16)
trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)
trainingLabelBitstrings = jnp.array(trainingLabelBitstrings)
initialPreds = apply_U(trainingPred, U)
accInitial = evaluate_classifier_top_k_accuracy(initialPreds, trainingLabel, 1)
costInitial = calculate_tanhCost(initialPreds, U, trainingLabelBitstrings, Zis)

U_init = np.copy(U) + 1e-3*np.random.randn(*U.shape)
U_update = jnp.array(U_init)
save_fname = 'tanh_jax_output.csv'

with open(save_fname, 'w') as f:
    line = 'pIndex, A, f0, decayRate, step, f, accuracy, cost\n'
    f.write(line)

def curr_f(decayRate, itNumber, initialRate):
    return initialRate / (1 + decayRate * itNumber)

for j in tqdm(range(parameters.shape[0]), desc=' P Loop', position=0):
    U_update = jnp.array(U_init)
    A, decayRate, f0 = parameters[j]
    data = np.array([j, A, f0, decayRate, 0, f0, accInitial, costInitial]).reshape(1, -1)
    save_output(save_fname, data)

    for i in tqdm(range(Nsteps), desc=' step', position=1, leave=False):
        #print(f'Update step {i+1}')
        f = curr_f(decayRate, i, f0)
        if f < 5e-4:
            f = 5e-4
        #print(f'   f: {f}')
        U_update, costs = update_U(trainingPredJ, U_update, trainingLabelBitstrings,
                f=f, costs=True, A=A, Zis=Zis)
        updatePreds = apply_U(trainingPred, np.array(U_update))
        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        #print(f'   Accuracy: {accUpdate}')
        #print(f'   Cost: ', costs)
        #print("")


        data = np.array([j, A, f0, decayRate, i+1, f, accUpdate, costs]).reshape(1, -1)
        save_output(save_fname, data)
