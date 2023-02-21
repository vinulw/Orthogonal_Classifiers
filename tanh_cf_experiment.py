'''
Experiment to tune hyperparameters for tanh cost function training.

We have the following hyperparameters to train:
    - f0 : [0.02, 0.2]
    - decayRate: [0.05, 0.5]
    - A: [10, 1000]

Wil have 5 steps for each of the parameters. Leading to 125 experiments in total
'''
from stacking_simple import load_data, evaluate_classifier_top_k_accuracy
from stacking_simple import labelsToBitstrings, apply_U, calculate_tanhCost
from stacking_simple import update_U
import numpy as np
from tqdm import tqdm

def create_hyperparameters():
    As = np.linspace(10, 1000, 5)
    decayRates = np.linspace(0.05, 0.5, 5)
    f0s = np.linspace(0.02, 0.2, 5)

    parameters = np.array(np.meshgrid(As, decayRates, f0s)).T.reshape(-1, 3)

    np.savetxt('hyperparameters.csv', parameters, delimiter=',')

def save_output(fname, data):
    with open(fname, 'a') as f:
        np.savetxt(f, data, delimiter=',')

parameters = np.loadtxt('hyperparameters.csv', delimiter=',')

np.random.seed(1)
prefix = "data_dropbox/mnist/"
trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

N = 1000
Nsteps = 200
outputCost = True
####
# Select training parameters
####
parameters = np.array([parameters[100]])

## Change the file name
save_fname = 'tanh_output_100_long_run.csv'

trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
          prefix + trainingLabelPath,
          N)

acc = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)

U = np.eye(16)
trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)
initialPreds = apply_U(trainingPred, U)
accInitial = evaluate_classifier_top_k_accuracy(initialPreds, trainingLabel, 1)
costInitial = calculate_tanhCost(initialPreds, U, trainingLabelBitstrings)

U_update = np.copy(U) + 1e-3*np.random.randn(*U.shape)

with open(save_fname, 'w') as f:
    line = 'pIndex, A, f0, decayRate, step, f, accuracy, cost\n'
    f.write(line)

def curr_f(decayRate, itNumber, initialRate):
    return initialRate / (1 + decayRate * itNumber)

for j in tqdm(range(parameters.shape[0]), desc=' P Loop', position=0):
    A, decayRate, f0 = parameters[j]
    data = np.array([j, A, f0, decayRate, 0, f0, accInitial, costInitial]).reshape(1, -1)
    save_output(save_fname, data)

    for i in tqdm(range(Nsteps), desc=' step', position=1, leave=False):
        #print(f'Update step {i+1}')
        f = curr_f(decayRate, i, f0)
        if f < 5e-4:
            f = 5e-4
        #print(f'   f: {f}')
        U_update, costs = update_U(trainingPred, U_update, trainingLabelBitstrings,
                f=f, costs=True, A=A)
        updatePreds = apply_U(trainingPred, U_update)
        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        #print(f'   Accuracy: {accUpdate}')
        #print(f'   Cost: ', costs)
        #print("")


        data = np.array([j, A, f0, decayRate, i+1, f, accUpdate, costs]).reshape(1, -1)
        save_output(save_fname, data)




