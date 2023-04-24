from opt_einsum import contract
from stacking_simple import generate_Zi, generate_CoeffArr
from functools import reduce

import numpy as np

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



def train_3_copy():

    now = datetime.now()
    now = now.strftime('%d%m%Y%H%M%S')

    np.random.seed(1)
    #prefix = "data_dropbox/mnist/"
    prefix = "data_dropbox/fashion_mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    N = 1000

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

if __name__=="__main__":
    from stacking_simple import calculate_ZOverlap
    import time
    states = np.random.rand(100, 2**4) + 1j*np.random.rand(100, 2**4)
    states /= np.linalg.norm(states, axis=1).reshape(100, 1)

    n_copies = 3
    label_start = 4*(n_copies-1) + 1
    print(type(label_start))
    states2 = np.array([copy_state(s, n_copies) for s in states])

    U = np.eye(2**(4*n_copies))
    print(2**(4*n_copies - label_start))
    Zi = generate_Zi(4*n_copies, label_start)

    print('Calculating old...')
    start = time.perf_counter()
    old_Z = calculate_ZOverlap(states2, U, Zi)
    end = time.perf_counter()
    print(f'Took {end - start}s')
    print()
    print('Calculating new...')
    start = time.perf_counter()
    new_Z = calculate_ZOverlap_n(states, n_copies, U, Zi)
    end = time.perf_counter()
    print(f'Took {end - start}s')

    print()
    print('Comparing old vs new...')
    print(np.linalg.norm(old_Z - new_Z))
    print(np.allclose(old_Z, new_Z))

