import numpy as np
from ncon import ncon
from math import floor

def evaluate_classifier_top_k_accuracy(predictions, y_test, k):
    top_k_predictions = [
        np.argpartition(image_prediction, -k)[-k:] for image_prediction in predictions
    ]
    results = np.mean([int(i in j) for i, j in zip(y_test, top_k_predictions)])
    return results


def generate_Zi(N, i):
    '''
    Generate Zi = III...Z...III operator

    Args
    ----
    N : Operator size in qubits
    i : Index of Z operator in qubits
    '''
    Ibef = np.eye(2**(i-1))
    Iaft = np.eye(2**(N-i))
    Z = np.array([[1., 0.], [0, -1.]], dtype=complex)

    Zi = ncon([Ibef, Z, Iaft], ((-1, -4), (-2, -5), (-3, -6)))
    Zi = Zi.reshape(2**N, 2**N)

    return Zi

def test_generate_Zi():
    I = np.eye(2)
    Z = np.array([[1., 0.], [0, -1.]], dtype=complex)

    N = 4
    i = 2

    IZII = np.kron(I, np.kron(Z, np.kron(I, I)))

    Zi = generate_Zi(N, i)
    assert(np.allclose(IZII, Zi))

    N = 3
    i = 1
    ZII = np.kron(Z, np.kron(I, I))

    Zi = generate_Zi(N, i)
    assert(np.allclose(ZII, Zi))

def labelsToBitstrings(labels, qNo):
    '''
    Convert the array of labels as ints to bitStrings.

    Note max label number is 256 currently.

    Args
    ----
    labels (ndarray int) : Arr of ints of size (N,) where N in the number of
            images. Ints should be in the set {0, ..., 2**qubitNo}.
    qNo : Number of qubits in label space

    Output
    ------
    labelBitstrings (ndarray int) : Arr of size (N, qubitNo) with a label for
            each image in bitstring form.
    '''
    assert max(labels) < 256, 'Max label number is 256'

    N = labels.shape[0]
    labels_ = labels.reshape(N, 1).astype(np.uint8)
    labelBitstrings = np.unpackbits(labels_, axis=1).astype(int)

    return labelBitstrings[:, 8-qNo:]

def test_labelsToBitstrings():
    labels = np.array([3, 4, 0, 3, 1, 1, 7, 2, 6, 6])

    correct = np.array([[0,1,1],
              [1,0,0],
              [0,0,0],
              [0,1,1],
              [0,0,1],
              [0,0,1],
              [1,1,1],
              [0,1,0],
              [1,1,0],
              [1,1,0]])

    labelBitstrings = labelsToBitstrings(labels, 3)

    assert np.allclose(labelBitstrings, correct)


def generate_CoeffArr(labelBitstrings, i):
    '''
    Generate coefficients for each label in labels for a given Zi specified by
    N and i.

    Args
    ----
    labelBitstrings (float) : Array of labels for shape (M,) with each label in
                            set {0, 1, ..., 2**N - 1} in bitstring form zero
                            padded to length of state space
    i : Index of Z operator in Zi = III...Z...III
    '''

    labelSubspace = labelBitstrings[:, i]

    return np.where(labelSubspace == 1, -1, 0) # == (-1) ** labelSubspace


def calculate_ZOverlap(ϕs, U, Zi):
    '''
    Perform overlap calculation , < ϕ | U* Zi U | ϕ > for each ϕ in ϕs

    Args
    ----
    ϕ : Array containing states of shape (N, n) where N is the number of states
        and n is the size of the state.
    U : Stacking unitary
    Zi : III...Z...III measurement operator
    '''

    Zmeasure = np.einsum('ij,kj,lk,lm,im->i',
                         np.conj(ϕs), np.conj(U), Zi, U, ϕs)
    return Zmeasure

def calculate_dOdV(ϕs, U, Zi):
    '''
    Perform dOdV calculation , < ϕ |=  =Zi U | ϕ > for each ϕ in ϕs. Will leave
    a matrix of shape U.shape

    Args
    ----
    ϕs : Array containing states of shape (N, n) where N is the number of states
        and n is the size of the state.
    U : Stacking unitary
    Zi : III...Z...III measurement operator
    '''

    dOdVs = np.einsum('ij,kl,lm,im->ikj',
                      np.conj(ϕs), Zi, U, ϕs)

    return dOdVs

def apply_U(ϕs, U):
    '''
    Appy the stacking unitary U to ϕ. Currently only works for 1 copy.

    Args
    ----
    ϕs : Array containing states of shape (N, n) where N is the number of states
        and n is the size of the state.
    U : Stacking unitary
    '''

    return np.einsum('lm,im->il', U, ϕs)


def update_U(ϕs, U, labelBitstrings, f=10):
    '''
    Do a single update of U using tanh cost function.

    Args
    ----
    ϕs : Array containing states of shape (N, n) where N is the number of states
        and n is the size of the state.
    U : Initial stacking unitary to update
    labelBitstrings (float) : Array of labels for ϕ in bitstring form zero
                            padded to length of state space in qubits as an
                            array of size (N, qubitNo) each element in this
                            array ∈ {0, 1}
    '''
    qNo = int(np.log2(ϕs.shape[1])) # No. qubits

    dZ = np.zeros(U.shape, dtype=complex)
    for i in range(qNo):
        #print(f"On Zi : {i}")
        Zi = generate_Zi(qNo, i+1)
        coeffArr = generate_CoeffArr(labelBitstrings, i)

        Zoverlaps = np.real(calculate_ZOverlap(ϕs, U, Zi))
        dOdVs = calculate_dOdV(ϕs, U, Zi)

        dZi = (np.cosh(Zoverlaps))**(-2)
        dZi = np.einsum('i,i,ijk->jk', coeffArr, Zoverlaps, dOdVs)

        dZ += dZi

    # Normalisation leads to instability
    dZ = dZ / (np.sqrt(ncon([dZ, dZ.conj()], [[1, 2], [1, 2]])) + 1e-14)

    U_update = get_Polar(dZ + f*U)

    return U_update

def get_Polar(M):
    """ Return the polar decomposition of M """
    from svd_robust import svd
    x,y,z =  svd(M)
    M = ncon([x,z],([-1,1],[1,-2]))
    return M

def load_data(statePath, labelPath, Nsamples=None):
    states = np.load(statePath)[15]
    labels = np.load(labelPath)

    if Nsamples is None:
        return states, labels

    noLabels = len(set(labels))
    samplesPerLabel = int(floor(Nsamples/noLabels))
    totalSamples = noLabels * samplesPerLabel
    stateSize = states.shape[1]

    outlabels = np.empty(totalSamples, dtype=int)
    outStates = np.empty((totalSamples, stateSize), dtype=states.dtype)

    for i, label in enumerate(set(labels)):
        labelArgs = np.argwhere(labels == label).flatten()
        sampleIndices = np.random.choice(labelArgs, [samplesPerLabel])

        outlabels[i*samplesPerLabel: (i+1)*samplesPerLabel] = i
        outStates[i*samplesPerLabel: (i+1)*samplesPerLabel] = states[sampleIndices]

        # # Debugging
        # print(i)
        # print('Checking index range...')
        # print(min(labelArgs))
        # print(max(labelArgs))
        # print('Checking correct labels...')
        # print(min(labels[labelArgs]))
        # print(max(labels[labelArgs]))
        # print('')

        assert(np.all(labels[labelArgs] == i))

    return outStates, outlabels


if __name__=="__main__":
    '''
    Use data from Lewis' dropbox
    '''
    prefix = "data_dropbox/mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    N = 1000

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

    acc = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)
    print(acc)

    U = np.eye(16)
    trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)

    initialPreds = apply_U(trainingPred, U)
    accInitial = evaluate_classifier_top_k_accuracy(initialPreds, trainingLabel, 1)

    U_update = update_U(trainingPred, U, trainingLabelBitstrings)

    updatePreds = apply_U(trainingPred, U_update)
    accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)

    print('Initial accuracy: ', accInitial)
    print('Update 0 accuracy: ', accUpdate)

    f = 5
    for i in range(10):
        print(f'Update step {i+1}')
        U_update = update_U(trainingPred, U_update, trainingLabelBitstrings, f=f)
        updatePreds = apply_U(trainingPred, U_update)
        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        print(f'Update {i+1} accuracy: {accUpdate}')


