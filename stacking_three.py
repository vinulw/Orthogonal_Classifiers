from opt_einsum import contract
from stacking_simple import generate_Zi, generate_CoeffArr
from functools import reduce

import numpy as np

def copy_state(ϕ, n):
    '''
    Copy a state n times such that |ϕ> → |ϕϕϕ...ϕ>
    '''
    return reduce(np.kron, [ϕ]*n)


def calculate_ZOverlap(ϕs, n, U, Zi):
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
    Zmeasures = np.zeros(ϕs.shape[0])
    for i in range(N):
        ϕ = ϕs[i]
        Zmeasure = contract('ij,kj,lk,lm,im->i',
                            np.conj(ϕs), np.conj(U), Zi, U, ϕs)
    return Zmeasures

def update_n_U(ϕs, U, labelBitstrings, n=3, f=0.1, costs=False, A=100, label_start=0):
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
    state1 = np.random.rand(2**4)

    state2 = np.kron(state1, state1)
    state2_red = copy_state(state1, 2)

    state3 = np.kron(state2, state1)
    state3_red = copy_state(state1, 3)

    print('Verifying 2 copy...')
    print(np.allclose(state2, state2_red))

    print('Verifying 3 copy...')
    print(np.allclose(state3, state3_red))
