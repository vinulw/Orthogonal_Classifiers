import numpy as np
from ncon import ncon
from math import floor
import matplotlib.pyplot as plt

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

def calculate_tanhCost(ϕs, U, labelBitstrings):
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
    '''
    qNo = int(np.log2(ϕs.shape[1])) # No. qubits
    N = ϕs.shape[0]

    totalCost = 0.0

    for i in range(qNo):
        Zi = generate_Zi(qNo, i+1)
        coeffArr = generate_CoeffArr(labelBitstrings, i)

        Zoverlaps = np.real(calculate_ZOverlap(ϕs, U, Zi))

        currCost = np.tanh(Zoverlaps)
        currCost = np.einsum('i,i', coeffArr, currCost) / (2*N)

        totalCost += currCost
    return totalCost


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

    return np.where(labelSubspace == 1, -1, 1) # == (-1) ** labelSubspace


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


def update_U(ϕs, U, labelBitstrings, f=0.1, costs=False):
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
    A = 10
    qNo = int(np.log2(ϕs.shape[1])) # No. qubits
    N = ϕs.shape[0]

    dZ = np.zeros(U.shape, dtype=complex)
    totalCost = 0.0
    for i in range(qNo):
        #print(f"On Zi : {i}")
        Zi = generate_Zi(qNo, i+1)
        coeffArr = generate_CoeffArr(labelBitstrings, i)

        Zoverlaps = np.real(calculate_ZOverlap(ϕs, U, Zi))
        dOdVs = calculate_dOdV(ϕs, U, Zi)

        dZi = A * (np.tanh(A)*np.cosh(A*np.sign(Zoverlaps)*np.absolute(Zoverlaps)))**(-2)
        dZi = np.einsum('i,i,ijk->jk', coeffArr, dZi, dOdVs)

        dZ += dZi

        if costs:
            currCost = np.tanh(Zoverlaps)
            currCost = np.einsum('i,i', coeffArr, currCost) / (2*N)
            totalCost += currCost

    # Normalisation leads to instability
    dZ = dZ / (np.sqrt(ncon([dZ, dZ.conj()], [[1, 2], [1, 2]])) + 1e-14)

    U_update = get_Polar(U + f*dZ)

    if costs:
        return U_update, totalCost
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
    np.random.seed(1)
    prefix = "data_dropbox/mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    N = 1000
    Nsteps = 50
    outputCost = True

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

    acc = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)
    print(acc)

    U = np.eye(16)
    trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)

    setTraininglabels = np.unique(trainingLabelBitstrings, axis=0)

    for i in range(4):
        print(f"Zi on {i}")
        coeffArr = generate_CoeffArr(setTraininglabels, i)

        for coeff, bstring in zip(coeffArr, setTraininglabels):
            print(f'{bstring} :  {coeff}')
        print("")

    initialPreds = apply_U(trainingPred, U)
    accInitial = evaluate_classifier_top_k_accuracy(initialPreds, trainingLabel, 1)
    costInitial = calculate_tanhCost(initialPreds, U, trainingLabelBitstrings)


    print('Initial accuracy: ', accInitial)
    print("")

    U_update = np.copy(U) + 1e-3*np.random.randn(*U.shape)

    f0 = 0.2
    f = np.copy(f0)
    decayRate = 0.3
    def curr_f(decayRate, itNumber, initialRate):
        return initialRate / (1 + decayRate * itNumber)

    costsList = [costInitial]
    accuracyList = [accInitial]
    fList = []
    for i in range(Nsteps):
        print(f'Update step {i+1}')
        f = curr_f(decayRate, i, f0)
        if f < 5e-4:
            f = 5e-4
        print(f'   f: {f}')
        U_update, costs = update_U(trainingPred, U_update, trainingLabelBitstrings,
                f=f, costs=True)
        updatePreds = apply_U(trainingPred, U_update)
        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        print(f'   Accuracy: {accUpdate}')
        print(f'   Cost: ', costs)
        print("")

        accuracyList.append(accUpdate)
        costsList.append(costs)
        fList.append(f)

    plt.figure()
    plt.title('Accuracy')
    plt.plot(accuracyList)

    plt.figure()
    plt.title('Costs')
    plt.plot(costsList)

    plt.figure()
    plt.title('Learning Rates')
    plt.plot(fList)

    plt.show()
    assert()
    # Trying scipy minimizer
    from scipy.optimize import minimize
    def costf(U):
        U = U.reshape(16, 16)
        cost = calculate_tanhCost(trainingPred, U, trainingLabelBitstrings)
        return -1*cost

    def callback(xk, state):
        print(f'Current iteration: {state.nit}')
        print(f'Current f: {state.fun}')

    res = minimize(costf, x0=U_update.reshape(16**2), method='Nelder-Mead',
                   options={'disp': True, 'maxiter': 20}, callback=callback)
    print(res)
    U = res.x .reshape(16, 16)

    updatePred = apply_U(trainingPred, U)
    accUpdate = evaluate_classifier_top_k_accuracy(updatePred, trainingLabel, 1)
    print("Trying scipy minimizer")
    print(f'Accuracy:  {accUpdate}')
    print("Applying polar decomposition...")
    U = get_Polar(U)
    updatePred = apply_U(trainingPred, U)
    accUpdate = evaluate_classifier_top_k_accuracy(updatePred, trainingLabel, 1)
    print(f'Accuracy:  {accUpdate}')







