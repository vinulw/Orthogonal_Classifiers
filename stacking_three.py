from opt_einsum import contract
from stacking_simple import generate_Zi, generate_CoeffArr
from stacking_simple import load_data
from functools import reduce
from datetime import datetime
from ncon import ncon
from collections.abc import Iterable

import numpy as np
import time

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

    #print('Calculating initial cost...')
    #start = time.perf_counter()
    #costInitial = calculate_tanhCost(states, U, trainingLabelBs, label_start= ls)
    #end = time.perf_counter()
    #print(f'Time taken: {end-start}s')
    #print(f'Initial cost: {costInitial}')


    print('Calculating old update U...')
    start = time.perf_counter()
    U_updated, cost = update_U(states, U, trainLabelBs, costs=True, label_start = ls)
    end = time.perf_counter()
    print(f'Time taken: {end - start}s')
    print()

    print('Calculating linear update U...')
    start = time.perf_counter()
    U_update_linear, cost_linear = update_U_linear(states, U, trainLabelBs, costs=True, label_start = ls)
    end = time.perf_counter()
    print(f'Time taken: {end - start}s')
    print()

    # print(f'Old cost {cost}')
    print(f'New cost {cost_linear}')
    print()

    print('Checking close...')
    print(np.allclose(U_updated, U_update_linear))
    print(np.linalg.norm(U_updated - U_update_linear))




if __name__=="__main__":
    train_3_copy()

