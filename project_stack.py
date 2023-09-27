import numpy as np
import stacking_simple as ss

def copy_state(ϕ, n):
    '''
    Copy a state n times such that |ϕ> → |ϕϕϕ...ϕ>
    '''
    return reduce(np.kron, [ϕ]*n)

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


if __name__=="__main__":
    np.random.seed(1)
    prefix = "data_dropbox/fashion_mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    ncopies = 2
    N = 1000
    maxiter = 10


    dim = 2**(4*ncopies)

    trainingPred, trainingLabel = ss.load_data(prefix + trainingPredPath,

    trainStates = np.array([copy_state(s, n_copies) for s in trainingPred])
    trainLabelBs = ss.labelsToBitstrings(trainingLabel, 4)


    U = np.eye(dim, dtype=complex)
    U_update = np.copy(U) + 1e-8*np.random.randn(*U.shape)

    predsInitial = pred_U_state(trainStates, U_update)
    accInitial = ss.evaluate_classifier_top_k_accuracy(predsInitial, trainingLabel, 1)

    for n in range(maxiter):


