from opt_einsum import contract
from stacking_simple import generate_Zi, generate_CoeffArr
from stacking_simple import load_data, get_Polar
from stacking_simple import apply_U
from stacking_simple import evaluate_classifier_top_k_accuracy
from functools import reduce
from datetime import datetime
from ncon import ncon
from collections.abc import Iterable

import os
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

def outer_sum(xs, ys, dim):
    sum = np.zeros((dim, dim), dtype=complex)
    for x, y in zip(xs, ys):
        sum += np.outer(x, y)
    return sum


def copy_state(ϕ, n):
    '''
    Copy a state n times such that |ϕ> → |ϕϕϕ...ϕ>
    '''
    return reduce(np.kron, [ϕ]*n)


def calculate_ZOverlap_n(ϕs, n, U, Zi):
    '''
    Perform overlap calculation , < ϕ | U* Zi U | ϕ > for each ϕ in ϕs

    Args
    ----
    ϕ : Array containing states of shape (N, qN) where N is the number of states
        and qN is the size of a single state.
    n : Number of copies of the state ϕ
    U : Stacking unitary
    Zi : III...Z...III measurement operator
    '''

    #Zmeasure = np.einsum('ij,kj,lk,lm,im->i',
    #                     np.conj(ϕs), np.conj(U), Zi, U, ϕs)
    N = ϕs.shape[0]
    Zmeasures = np.zeros(ϕs.shape[0], dtype=complex)
    for i in range(N):
        ϕ = ϕs[i]
        ϕ = copy_state(ϕ, n)
        Zmeasure = contract('j,kj,lk,lm,m',
                            np.conj(ϕ), np.conj(U), Zi, U, ϕ)
        Zmeasures[i] = Zmeasure
    return Zmeasures

def update_U_linear(ϕs, U, labelBitstrings, f=0.1, costs=False, A=100, label_start=0):
    '''
    Do a single update of U using tanh cost function.

    Args
    ----
    ϕs : Array containing states of shape (N, n) where N is the number of states
        and n is the size of a single copy of the state.
    U : Initial stacking unitary to update
    labelBitstrings (float) : Array of labels for ϕ in bitstring form zero
                            padded to length of state space in qubits as an
                            array of size (N, qubitNo) each element in this
                            array ∈ {0, 1}
    '''
    from stacking_simple import calculate_ZOverlap
    A_iter = False
    if isinstance(A, Iterable):
        A_iter = True
        A_curr = A[0]
    else:
        A_curr = A

    qNo = int(np.log2(ϕs.shape[1])) # No. qubits
    N = ϕs.shape[0]
    dim = ϕs.shape[1]

    dZ = np.zeros(U.shape, dtype=complex)
    totalCost = 0.0

    Ustates = np.einsum('lm, im -> il', U, ϕs)

    out = np.empty(U.shape, dtype=complex) # Hold dOdV, speed up outer
    ϕs_dag = np.conj(ϕs)
    for i in tqdm(range(qNo - label_start), total=qNo-label_start):
        # print(f'   Current qi: {i}')
        if A_iter:
            A_curr = A[i]
        Zi = generate_Zi(qNo, i+1+label_start)
        Zoverlaps = np.real(calculate_ZOverlap(ϕs, U, Zi))
        dZi = A_curr * (1 - np.tanh(A_curr*Zoverlaps)**2)

        coeffArr = generate_CoeffArr(labelBitstrings, i)
        # Iterate over each state to get contribution to update
        dZiUstate = ncon([Zi, Ustates], ((-2, 1), (-1, 1)))
        coeffArrdZi = coeffArr * dZi
        coeffArrdZiUstate = coeffArrdZi.reshape(-1, 1) * dZiUstate
        for j in range(N):
            dZ += np.outer(coeffArrdZiUstate[j], ϕs_dag[j], out)

        if costs:
            currCost = np.tanh(A_curr*Zoverlaps)
            currCost = np.einsum('i,i', coeffArr, currCost) / N
            totalCost += currCost

    # Normalisation leads to instability
    dZ = dZ / (np.sqrt(ncon([dZ, dZ.conj()], [[1, 2], [1, 2]])) + 1e-14)

    #U_update = get_Polar(U + f*dZ)
    U_update = U + f*dZ
    U_update = U_update / np.linalg.norm(U_update)

    if costs:
        return U_update, totalCost
    return U_update

def trace_states(states, partition_index):
    '''
    Trace (N, d) array representing N d-dimensional states about a partition_index.
    Note the traced segment is always the one before the partition.
    '''
    N, d = states.shape

    states_ = states.reshape(N, partition_index, -1)
    return contract('ijk, ijl -> ikl', states_, states_.conj())

def pred_U_state(states, U, labelNo=4):
    dim = states.shape[1]
    statesU = apply_U(states, U)
    partition_index = dim // 2**labelNo
    rhoU = trace_states(statesU, partition_index)
    return np.diagonal(rhoU, axis1=1, axis2=2)

def train_3_copy(config_path, U0=None, save=False, save_interval=10):
    from stacking_simple import calculate_tanhCost, labelsToBitstrings
    from stacking_simple import update_U
    now = datetime.now()
    now = now.strftime('%d%m%Y%H%M%S')

    np.random.seed(1)
    #prefix = "data_dropbox/mnist/"
    prefix = "data_dropbox/fashion_mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    # Load the config file
    with open(config_path, 'r') as cf:
        config_dict = json.load(cf)

    n_copies = config_dict['Ncopies']

    N = config_dict.get("Ntrain", 1000)
    n_copies = config_dict.get("Ncopies", 2)
    dim = 2**(4*n_copies)
    ls = 4*(n_copies - 1)

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

    trainStates = np.array([copy_state(s, n_copies) for s in trainingPred])
    trainLabelBs = labelsToBitstrings(trainingLabel, 4)
    if U0 is None:
        U = np.eye(dim, dtype=complex)
        U_update = np.copy(U) + 1e-12*np.random.randn(*U.shape)
    else:
        U_update = np.copy(embed_U(U0, n_copies*4))

    # Fashion MNIST 2 copy
    As = config_dict["As"]
    Ai = 0
    switch_index = config_dict["switch_index"]
    Nsteps = config_dict["Nsteps"]

    # Fashion MNIST
    f0 = config_dict["f0"]
    fmin = config_dict.get("fmin", 0.02)
    f = np.copy(f0)
    decayRate = config_dict["decayRate"]
    ortho_step = config_dict.get('orthoStep', Nsteps + 10)

    def curr_f(decayRate, itNumber, initialRate):
        return initialRate / (1 + decayRate * itNumber)

    costInitial = calculate_tanhCost(trainStates, U_update, trainLabelBs, label_start= ls, A=As[Ai])
    predsInitial = pred_U_state(trainStates, U_update)
    accInitial = evaluate_classifier_top_k_accuracy(predsInitial, trainingLabel, 1)

    print('Initial accuracy: ', accInitial)
    print('Initial cost: ', costInitial)
    print("")

    costsList = [costInitial]
    accuracyList = [accInitial]
    fList = []
    # ortho_step = Nsteps + 10
    i = 0

    if save:
        save_dir = f'tanh_{n_copies}_copy/{now}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f'Made save directory: {save_dir}')
        classifier_dir = save_dir + 'classifier_U/'
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)
            print(f'Made classifier directory: {classifier_dir}')
        # CSV file to track training
        csv_data_file = save_dir + 'run_data.csv'

        with open(csv_data_file, 'w') as f:
            header = 'accuracy, cost, lr'
            line = np.array([accInitial, costInitial, f0])
            np.savetxt(f, line.reshape(1, -1), delimiter=', ', header=header)

    for n in range(Nsteps):
        dRate = decayRate[Ai]
        start = time.perf_counter()
        A = As[Ai]
        print(f'Update step {n+1}')
        f = curr_f(dRate, i, f0)
        if f < fmin:
            f = fmin
        print(f'   f: {f}')
        if n_copies < 3:
            U_update, costs = update_U(trainStates, U_update, trainLabelBs,
                    f=f, costs=True, A=A, label_start=ls)
        else:
            U_update, costs = update_U_linear(trainStates, U_update, trainLabelBs,
                    f=f, costs=True, A=A, label_start=ls)

        if i != 0 and i % ortho_step == 0:
            print('Polarising...')
            U_update = get_Polar(U_update)

        updatePreds = pred_U_state(trainStates, U_update)

        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        print(f'   Accuracy: {accUpdate}')
        print(f'   Cost: ', costs)

        accuracyList.append(accUpdate)
        costsList.append(costs)
        fList.append(f)

        # For running with Fashion MNIST
        if n in switch_index:
            if save:
                save_name = save_dir + f'step_{n}_hist.png'
                classifier_name = classifier_dir + f'step_{n}.npy'
                np.save(classifier_name, U_update)
            print('Resetting Ai and f0')
            Ai += 1
            f0 = f0*0.8
            i = 0

        if save:
            with open(csv_data_file, 'a') as fle:
                line = np.array([accUpdate, costs, f])
                np.savetxt(fle, line.reshape(1, -1), delimiter=', ')
            if n % save_interval == 0:
                save_name = save_dir + f'step_{n}_hist.png'
                classifier_name = classifier_dir + f'step_{n}.npy'
                np.save(classifier_name, U_update)

        i += 1

        end = time.perf_counter()

        print(f'Step elapsed time: {end - start:0.4f} seconds')
        print("")

    if save:
        classifier_name = classifier_dir + f'step_{Nsteps}.npy'
        np.save(classifier_name, U_update)

    plt.figure()
    plt.title('Accuracy')
    plt.plot(accuracyList)
    plt.savefig(save_dir + 'accuracy.png')

    plt.figure()
    plt.title('Costs')
    plt.plot(costsList)
    plt.savefig(save_dir + 'costs.png')

    plt.figure()
    plt.title('Learning Rates')
    plt.plot(fList)
    plt.savefig(save_dir + 'lr.png')

    plt.show()

def embed_U(U, qNo):
    '''
    Embed U representing a stacking unitary on qOld qubits into a new U_prime acting on qNo qubits.
    This embedding is done so that U_prime = I ⊗ U.
    '''
    dim_old = U.shape[0]
    dim_new = 2**qNo
    if dim_old == dim_new:
        return U
    assert dim_new > dim_old
    I = np.eye(dim_new // dim_old, dtype=complex)

    U_prime = np.kron(I, U)

    return U_prime


if __name__=="__main__":
    print('Loading U0...')
    # U0 = np.load('tanh_2_copy/01052023153003/classifier_U/step_150.npy', allow_pickle=True)
    # prefix = '/home/ucapvdw/Projects/project-orthogonal_classifiers/tanh_data/'
    # U0 = np.load(prefix + '01052023181353/classifier_U/step_140.npy', allow_pickle=True)
    prefix = "/mnt/c/Users/vwimalaweera/OneDrive - University College London/Project_Files/project-orthogonal_classifier/tanh_2_copy/06062023160233/classifier_U/"
    U0 = np.load(prefix + 'step_70.npy')
    print('Doing polar decomposition...')
    U0 = get_Polar(U0)
    print('Training...')
    train_3_copy("experiment_param_two.json", U0=U0, save=True)

