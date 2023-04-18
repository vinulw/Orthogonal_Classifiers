import numpy as np
from ncon import ncon
from math import floor
import matplotlib.pyplot as plt
from collections.abc import Iterable
from opt_einsum import contract
from tqdm import tqdm

import os
from datetime import datetime
import time

from stacking_simple import generate_Zi, generate_CoeffArr, load_data, calculate_ZOverlap
from stacking_simple import evaluate_classifier_top_k_accuracy
from stacking_simple import labelsToBitstrings
from stacking_simple import trace_rho, apply_U_rho, get_Polar


def calculate_tanhCost(ϕs, U, labelBitstrings, A=[1], label_start=0):
    '''
    Calculate total tanh Cost

    Args
    ----
    ϕs : Array containing states of shape (N, n) where N is the number of states
        and n is the size of the state.
    U : Initial stacking unitary to update
    labelBitstrings (float) : Array of labels for ϕ in bitstring form zero
                            padded to length of state space in qubits as an
                            array of size (N, qubitNo) each element in this
                            array ∈ {0, 1}
    label_start : Start index for the label qubits
    '''
    qNo = int(np.log2(ϕs.shape[1])) # No. qubits
    N = ϕs.shape[0]

    totalCost = 0.0

    for i in range(qNo - label_start):
        if len(A) > 1:
            A_curr = A[i]
        else:
            A_curr = A[0]
        Zi = generate_Zi(qNo, i+1+label_start)
        coeffArr = generate_CoeffArr(labelBitstrings, i)

        Zoverlaps = np.real(calculate_ZOverlap(ϕs, U, Zi))

        currCost = np.tanh(A_curr * Zoverlaps)
        currCost = np.einsum('i,i', coeffArr, currCost) / (N)

        totalCost += currCost
    return totalCost

def train_stochastic():
    # Fashion MNIST 2 copy
    As = [[500, 500, 500, 500],
          [5000, 5000, 5000, 5000]]
    Ai = 0

    now = datetime.now()
    now = now.strftime('%d%m%Y%H%M%S')

    np.random.seed(1)
    #prefix = "data_dropbox/mnist/"
    prefix = "data_dropbox/fashion_mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    save_dir = f'stochastic_train/{now}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'Made save directory: {save_dir}')
    show = False

    N = 1000

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

    acc = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)
    print('Initial accuracy: ', acc)

    trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)

    # Make two copies
    Ncopies = 1
    label_start = 0
    qNo = 4*Ncopies
    dim_N = 2**qNo

    if Ncopies == 2:
        trainingPred = np.array([np.kron(im, im) for im in trainingPred])
    ρPred = np.array([np.outer(pred, pred.conj()) for pred in trainingPred])
    print(ρPred.shape)

    U = np.eye(dim_N)
    U = U / np.linalg.norm(U)
    U += 1e-12*np.random.randn(*U.shape)

    initialrho = apply_U_rho(ρPred, U)
    if Ncopies == 2:
        initialrho_ = trace_rho(initialrho, qNo, trace_ind=[0, 1, 2, 3])
    else:
        initialrho_ = trace_rho(initialrho, qNo, trace_ind=[])
    initialPreds = np.diagonal(initialrho_, axis1=1, axis2=2) # Get Tr(Pi ρ)

    accInitial = evaluate_classifier_top_k_accuracy(initialPreds, trainingLabel, 1)
    costInitial = calculate_tanhCost(trainingPred, U, trainingLabelBitstrings, A=As[0], label_start=label_start)

    print('Initial accuracy: ', accInitial)
    print('Initial cost: ', costInitial)
    print("")


    start = time.perf_counter()

    # Saving classifiers
    save_interval = 10
    classifier_dir = save_dir + 'classifier_U/'
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)
        print(f'Made classifier directory: {classifier_dir}')

    # Saving classifiers
    save_interval = 10
    classifier_dir = save_dir + 'classifier_U/'
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)
        print(f'Made classifier directory: {classifier_dir}')

    # CSV file to track training
    csv_data_file = save_dir + 'run_data.csv'

    with open(csv_data_file, 'w') as f:
        header = 'accuracy, cost, lr'
        line = np.array([accInitial, costInitial])
        np.savetxt(f, line.reshape(1, -1), delimiter=', ', header=header)


    costsList = [costInitial]
    accuracyList = [accInitial]

    Nsteps = 10000
    ortho_step = Nsteps + 10

    alpha = 0.1
    p = 0.0

    U_curr = np.copy(U)
    acc_curr = accInitial
    for n in tqdm(range(Nsteps), total=Nsteps):
        A = As[Ai]

        U_ = get_Polar(U_curr + alpha*np.random.rand(*U_curr.shape))

        Cold = costsList[-1]
        Cnew = calculate_tanhCost(trainingPred, U_,
                trainingLabelBitstrings, label_start=label_start)

        if Cold < Cnew or p > np.random.rand(1):
            #print(f'Optimised C step {n}')
            U_curr = np.copy(U_)
            rhoU = apply_U_rho(ρPred, U_curr)
            if Ncopies == 2:
                rhoU = trace_rho(rhoU, qNo, trace_ind=[0, 1, 2, 3])
            else:
                rhoU = trace_rho(rhoU, qNo, trace_ind=[])
            predsU = np.diagonal(rhoU, axis1=1, axis2=2) # Get Tr(Pi ρ)
            acc_curr = evaluate_classifier_top_k_accuracy(predsU, trainingLabel, 1)
            #print(f'New accuracy: {acc_curr}')
            accuracyList.append(acc_curr)
            costsList.append(Cnew)
        else:
            accuracyList.append(acc_curr)
            costsList.append(Cold)

    print(f'Final cost: {costsList[-1]}')
    print(f'Final accuracy: {accuracyList[-1]}')



    plt.plot(costsList)
    plt.title('Costs')
    plt.figure()
    plt.plot(accuracyList)
    plt.title('Accuracy')
    plt.show()


if __name__=="__main__":
    train_stochastic()

