from opt_einsum import contract
from stacking_simple import generate_Zi, generate_CoeffArr
from stacking_simple import load_data
from functools import reduce
from datetime import datetime

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

def update_U_n(ϕs, U, labelBitstrings, n=3, f=0.1, costs=False, A=100, label_start=0):
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
    n : Number of copies of ϕ to take
    '''
    A_iter = False
    if isinstance(A, Iterable):
        A_iter = True
        A_curr = A[0]
    else:
        A_curr = A

    N = ϕs.shape[0]
    labelNo = labelBitstrings.shape[1] # Number of label qubits

    qNo = int(np.log2(ϕs.shape[1])) # No. qubits
    qNo *= n

    for i in range(labelNo):
        if A_iter:
            A_curr = A[i]
        Zi = generate_Zi(qNo, i+1+label_start)
        coeffArr = generate_CoeffArr(labelBitstrings, i)

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
    trainingLabelBs = labelsToBitstrings(trainingLabel, 4)
    U = np.eye(dim, dtype=complex)

    #print('Calculating initial cost...')
    #start = time.perf_counter()
    #costInitial = calculate_tanhCost(states, U, trainingLabelBs, label_start= ls)
    #end = time.perf_counter()
    #print(f'Time taken: {end-start}s')
    #print(f'Initial cost: {costInitial}')

    print('Now modifying the way U is applied')
    from stacking_simple import apply_U_rho, trace_rho
    print('Calculating using old method...')
    start = time.perf_counter()
    qNo = 4*n_copies
    rhos =  np.array([np.outer(s, s.conj()) for s in states])
    rhos = apply_U_rho(rhos, U)
    traced_rhos = trace_rho(rhos, qNo, trace_ind=[0, 1, 2, 3])
    end = time.perf_counter()
    print(f'Time taken: {end-start}s')

    print('Applying U linearly...')
    from stacking_simple import apply_U
    start = time.perf_counter()
    statesU = apply_U(states, U)
    traced_rho_new = np.zeros(traced_rhos.shape, dtype=complex)
    for i, s in enumerate(statesU):
        s_ = s.reshape(-1, 2**4)
        rho_s_ = contract('ij, ik -> jk', s_, s_.conj())
        traced_rho_new[i] = np.copy(rho_s_)
    end = time.perf_counter()
    print(f'Time taken: {end-start}s')

    print('Applying U to state...')
    start = time.perf_counter()
    statesU = apply_U(states, U)
    N = statesU.shape[0]
    statesU = statesU.reshape(N, -1, 2**4)
    print(statesU.shape)
    trace_rho_applied = contract('ijk, ijl -> ikl', statesU, statesU.conj())
    end = time.perf_counter()
    print(f'Time taken: {end-start}s')

    print('ptrace Function...')
    start = time.perf_counter()
    statesU = apply_U(states, U)
    dim = statesU.shape[1]
    partition_index = dim // (2**4)
    trace_rho_func = trace_states(statesU, partition_index)
    end = time.perf_counter()
    print(f'Time taken: {end-start}s')


    print('Checking close to linear...')
    print(np.allclose(traced_rhos[0], traced_rho_new[0]))
    print(np.linalg.norm(traced_rhos - traced_rho_new))

    print('Checking close to applied...')
    print(np.allclose(traced_rhos[0], trace_rho_applied[0]))
    print(np.linalg.norm(traced_rhos - trace_rho_applied))

    print('Checking close to func...')
    print(np.allclose(traced_rhos[0], trace_rho_func[0]))
    print(np.linalg.norm(traced_rhos - trace_rho_func))










if __name__=="__main__":
    train_3_copy()

