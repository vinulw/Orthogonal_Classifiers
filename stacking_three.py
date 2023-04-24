from opt_einsum import contract
from stacking_simple import generate_Zi, generate_CoeffArr
from stacking_simple import load_data, get_Polar
from stacking_simple import apply_U
from stacking_simple import evaluate_classifier_top_k_accuracy
from functools import reduce
from datetime import datetime
from ncon import ncon
from collections.abc import Iterable

import numpy as np
import time
import matplotlib.pyplot as plt

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

    dZ = np.zeros(U.shape, dtype=complex)
    totalCost = 0.0

    for i in range(qNo - label_start):
        if A_iter:
            A_curr = A[i]
        Zi = generate_Zi(qNo, i+1+label_start)
        Zoverlaps = np.real(calculate_ZOverlap(ϕs, U, Zi))
        dZi = A_curr * (1 - np.tanh(A_curr*Zoverlaps)**2)

        coeffArr = generate_CoeffArr(labelBitstrings, i)
        # Iterate over each state to get contribution to update
        for j in range(N):
            state = ϕs[j]
            dOdV = contract('j, kl, lm, m ->kj',
                    np.conj(state), Zi, U, state)
            dZ += coeffArr[j] * dZi[j] * dOdV

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

def train_3_copy():
    from stacking_simple import calculate_tanhCost, labelsToBitstrings
    from stacking_simple import update_U
    now = datetime.now()
    now = now.strftime('%d%m%Y%H%M%S')

    np.random.seed(1)
    #prefix = "data_dropbox/mnist/"
    prefix = "data_dropbox/fashion_mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    N = 100
    n_copies = 2
    dim = 2**(4*n_copies)
    ls = 4*(n_copies - 1) + 1

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

    states = np.array([copy_state(s, n_copies) for s in trainingPred])
    trainLabelBs = labelsToBitstrings(trainingLabel, 4)
    U = np.eye(dim, dtype=complex)


    U_update = np.copy(U) + 1e-12*np.random.randn(*U.shape)

    # Fashion MNIST 2 copy
    As = [[500, 500, 500, 500],
          [5000, 5000, 5000, 5000]]
    Ai = 0
    switch_index = [50]
    Nsteps = 300

    # Fashion MNIST
    f0 = 0.10
    f = np.copy(f0)
    decayRate = 0.035

    def curr_f(decayRate, itNumber, initialRate):
        return initialRate / (1 + decayRate * itNumber)

    costInitial = calculate_tanhCost(states, U_update, trainLabelBs, label_start= ls, A=As[Ai])
    predsInitial = pred_U_state(states, U_update)
    accInitial = evaluate_classifier_top_k_accuracy(predsInitial, trainingLabel, 1)

    print('Initial accuracy: ', accInitial)
    print('Initial cost: ', costInitial)
    print("")

    costsList = [costInitial]
    accuracyList = [accInitial]
    fList = []
    ortho_step = Nsteps + 10
    i = 0

    start = time.perf_counter()
    print(ls)
    assert()
    for n in range(Nsteps):
        A = As[Ai]
        print(f'Update step {n+1}')
        f = curr_f(decayRate, i, f0)
        if f < 2e-3:
            f = 2e-3
        print(f'   f: {f}')
        U_update, costs = update_U(trainingPred, U_update, trainLabelBs,
                f=f, costs=True, A=A, label_start=ls)

        updatePreds = pred_U_state(states, U_update)

        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        print(f'   Accuracy: {accUpdate}')
        print(f'   Cost: ', costs)
        print("")

        accuracyList.append(accUpdate)
        costsList.append(costs)
        fList.append(f)

        # with open(csv_data_file, 'a') as fle:
        #     line = np.array([accUpdate, costs, f])
        #     np.savetxt(fle, line.reshape(1, -1), delimiter=', ')

        # For running with Fashion MNIST
        if n in switch_index:
            print('Resetting Ai and f0')
            Ai += 1
            f0 = f0*0.8
            i = 0

        # if n % save_interval == 0:
        #     save_name = save_dir + f'step_{n}_hist.png'
        #     plot_qubit_histogram(updatePreds, trainingLabelBitstrings,
        #             title=f'Step: {n}', show=False, save_name=save_name)
        #     classifier_name = classifier_dir + f'step_{n}.npy'
        #     np.save(classifier_name, U_update)

        if n % ortho_step:
            U_update = get_Polar(U_update)

        i += 1



    end = time.perf_counter()

    print(f'Elapsed time: {end - start:0.4f} seconds')

    plt.figure()
    plt.title('Accuracy')
    plt.plot(accuracyList)
    #plt.savefig(save_dir + 'accuracy.png')

    plt.figure()
    plt.title('Costs')
    plt.plot(costsList)
    #plt.savefig(save_dir + 'costs.png')

    plt.figure()
    plt.title('Learning Rates')
    plt.plot(fList)
    #plt.savefig(save_dir + 'lr.png')

    plt.show()



if __name__=="__main__":
    train_3_copy()

