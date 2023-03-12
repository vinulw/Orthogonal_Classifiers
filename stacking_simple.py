import numpy as np
from ncon import ncon
from math import floor
import matplotlib.pyplot as plt
from collections.abc import Iterable

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

def calculate_tanhCost(ϕs, U, labelBitstrings, A=1, label_start=0):
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
    print(qNo)
    N = ϕs.shape[0]

    totalCost = 0.0

    # TODO : This loop is taking ages....
    for i in range(qNo):
        print(i+1+label_start)
        Zi = generate_Zi(qNo, i+1+label_start)
        coeffArr = generate_CoeffArr(labelBitstrings, i)

        Zoverlaps = np.real(calculate_ZOverlap(ϕs, U, Zi))

        currCost = np.tanh(A * Zoverlaps)
        currCost = np.einsum('i,i', coeffArr, currCost) / (N)

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

def apply_U_rho(ρs, U):

    out = ncon([U, ρs, U.conj()], ((-2, 1), (-1, 1, 2), (-3, 2)))

    # To get the Tr(Pi U ρ U_dagger)
    # out = np.array([np.sqrt(np.diag(o)) for o in out])
    return out

def trace_rho(rho, qNo, trace_ind):
    N = rho.shape[0]
    rho = rho.reshape(N, *[2]*(qNo*2))
    contr_string = [-2-i for i in range(qNo*2)]
    curr_contr = 1
    for ind in trace_ind:
        contr_string[ind] = curr_contr
        contr_string[qNo + ind] = curr_contr
        curr_contr += 1

    contr_string = [-1] + contr_string
    rho = ncon([rho,], (contr_string,))

    rho_shape = int(2**((len(rho.shape) - 1)/2))
    rho = rho.reshape(N, rho_shape, rho_shape)

    return rho




def update_U(ϕs, U, labelBitstrings, f=0.1, costs=False, A=100, label_start=0):
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
    for i in range(qNo):
        #print(f"On Zi : {i}")
        if A_iter:
            A_curr = A[i]

        Zi = generate_Zi(qNo, i+1+label_start)
        coeffArr = generate_CoeffArr(labelBitstrings, i)

        Zoverlaps = np.real(calculate_ZOverlap(ϕs, U, Zi))
        dOdVs = calculate_dOdV(ϕs, U, Zi)

        #dZi = A * (np.tanh(A)*np.cosh(A*np.sign(Zoverlaps)*np.absolute(Zoverlaps)))**(-2)
        #dZi = A * (np.tanh(A)*np.cosh(A*Zoverlaps))**(-2)
        #dZi = A * (np.cosh(A*Zoverlaps))**(-2)
        dZi = A_curr * (1 - np.tanh(A_curr*Zoverlaps)**2)
        dZi = np.einsum('i,i,ijk->jk', coeffArr, dZi, dOdVs)

        dZ += dZi

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

def experiment_Zi_distances():
    np.random.seed(1)
    prefix = "data_dropbox/mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    N = 1000

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

    trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)

    perfectPred = np.zeros(trainingPred.shape)
    for i, label in enumerate(trainingLabel):
        perfectPred[i, label] = 1.

    qNo = 4
    U = np.eye(16)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for i in range(qNo):
        print(f'i: {i}')
        Zi = generate_Zi(qNo, i+1)

        ZiOverlaps = np.real(calculate_ZOverlap(trainingPred, U, Zi))
        coeffArr = generate_CoeffArr(trainingLabelBitstrings, i)

        Zi_minus = ZiOverlaps[coeffArr == -1]
        Zi_plus = ZiOverlaps[coeffArr == 1]

        print(Zi_minus.shape)
        print(Zi_plus.shape)

        ax = axs[i]
        ax.hist(Zi_minus, alpha=0.5)
        ax.hist(Zi_plus, color='r', alpha=0.5)
        ax.set_title(f'i: {i} ')

    plt.tight_layout(pad=2.0, w_pad=5., h_pad=10.0)
    plt.show()


def train_2_copy():
    import time

    np.random.seed(1)
    prefix = "data_dropbox/mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    N = 1000
    Nsteps = 20
    outputCost = True

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)


    #print(trainingPred.shape)

    #print(trainingPred[0])
    #print(np.linalg.norm(trainingPred[0]))

    perfectPred = np.zeros(trainingPred.shape)
    for i, label in enumerate(trainingLabel):
        perfectPred[i, label] = 1.
    #print(perfectPred[:5])

    acc = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)
    print('Initial accuracy: ', acc)

    acc = evaluate_classifier_top_k_accuracy(perfectPred, trainingLabel, 1)
    #print(f"Perfect prediction: {acc}")


    trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)

    # Make two copies
    qNo = 8
    dim_N = 2**qNo
    trainingPred = np.array([np.kron(im, im) for im in trainingPred])
    ρPred = np.array([np.outer(pred, pred.conj()) for pred in trainingPred])
    print(ρPred.shape)
    U = np.eye(dim_N)
    U = U / np.linalg.norm(U)

    initialrho = apply_U_rho(ρPred, U)
    initialPreds = trace_rho(initialrho, qNo, trace_ind=[0, 1, 2, 3])
    initialPreds = np.diagonal(initialPreds, axis1=1, axis2=2) # Get Tr(Pi ρ)

    accInitial = evaluate_classifier_top_k_accuracy(initialPreds, trainingLabel, 1)
    costInitial = calculate_tanhCost(trainingPred, U, trainingLabelBitstrings, label_start=4)

    print('Initial accuracy: ', accInitial)
    print('Initial cost: ', costInitial)
    print("")
    assert()

    perfectPred = apply_U(perfectPred, U)
    accPerfect = evaluate_classifier_top_k_accuracy(perfectPred, trainingLabel, 1)
    costPerfect = calculate_tanhCost(perfectPred, U, trainingLabelBitstrings)

    print("Perfect acc: ", accPerfect)
    print("Perfect cost: ", costPerfect)

    #trainingPred = perfectPred
    #costInitial = costPerfect
    #accInitial = accPerfect

    U_update = np.copy(U) + 1e-12*np.random.randn(*U.shape)

    start = time.perf_counter()


    A = 100
    A = [100, 10, 10, 10]
    # A = [10, 100, 100, 100] is my guess for the best results but not sure
    f0 = 0.15
    f = np.copy(f0)
    decayRate = 0.2
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
                f=f, costs=True, A=A)
        updatePreds = apply_U(trainingPred, U_update)
        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        print(f'   Accuracy: {accUpdate}')
        print(f'   Cost: ', costs)
        print("")

        accuracyList.append(accUpdate)
        costsList.append(costs)
        fList.append(f)

    end = time.perf_counter()

    print(f'Elapsed time: {end - start:0.4f} seconds')

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




if __name__=="__main__":
    train_2_copy()
    '''
    Use data from Lewis' dropbox
    '''
    from scipy.stats import unitary_group
    from functools import reduce



    N = 10
    qNo = 2
    dim_U = 2**qNo
    U = unitary_group.rvs(dim_U)
    vecs = np.random.rand(N, dim_U)
    norm = np.linalg.norm(vecs, axis=1)
    norm = np.repeat(norm[:, None], dim_U, axis=0)
    norm = norm.reshape(N, dim_U)
    vecs = vecs / norm

    ρ = np.array([np.outer(vec, vec.conj()) for vec in vecs])

    rho0 = ρ[0]
    vec0 = vecs[0]
    print('Vec0 == ρ0: ', np.allclose(vecs[0]**2, np.diag(rho0)))

    Urho0_ncon = ncon([U, rho0, U.conj()], ((-1, 1) ,(1, 2), (-2, 2)))
    Urho0_einsum= np.einsum('lm, mk, nk -> ln', U, rho0, U.conj())
    print('Ncon == einsum :', np.allclose(Urho0_ncon, Urho0_einsum))

    out = apply_U(vecs, U)
    out0 =  out[0]
    Uvec0 = ncon([U, vec0], ((-1, 1), (1,)))
    Uvec0conj = ncon([vec0.conj(), U.conj()], ((1,), (-1, 1)))
    print('U@v0 == out0 :', np.allclose(U@vec0, out0))
    print('Uv0 == out0 :', np.allclose(Uvec0, out0))
    print('Uvec0conj == out0.conj', np.allclose(Uvec0conj, out0.conj()))

    out0_rho = np.outer(out0, out0.conj())
    Uvec0_rho = np.outer(Uvec0, Uvec0conj)

    print('Uvec0_rho == out0_rho', np.allclose(out0_rho, Uvec0_rho))

    Urho0 = ncon([U, vec0, vec0.conj(), U.conj()], ((-1, 1), (1,), (2,), (-2, 2)))

    print('Out0_rho == Uρ0: ', np.allclose(out0_rho, Urho0))
    print('Out0_rho == Uρ0_ncon: ', np.allclose(out0_rho, Urho0_ncon))

    print(out0)
    print(out0*out0.conj())
    print(np.diag(out0_rho))
    print(np.diag(Urho0))
    print('Out0 == Out0_rho', np.allclose(out0*out0.conj(), np.diag(out0_rho)))
    print('Out0 == ρ0 :', np.allclose(out0*out0.conj(), np.diag(Urho0)))

    Uρ = apply_U_rho(ρ, U, qNo)

    print('Vecrtorised Urho = Urho0 :', np.allclose(Uρ[0], Urho0))

    assert()

    #experiment_Zi_distances()

    import time

    np.random.seed(1)
    prefix = "data_dropbox/mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    N = 1000
    Nsteps = 20
    outputCost = True

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)


    print(trainingPred.shape)

    print(trainingPred[0])
    print(np.linalg.norm(trainingPred[0]))

    perfectPred = np.zeros(trainingPred.shape)
    for i, label in enumerate(trainingLabel):
        perfectPred[i, label] = 1.
    print(perfectPred[:5])

    acc = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)
    print(acc)

    acc = evaluate_classifier_top_k_accuracy(perfectPred, trainingLabel, 1)
    print(f"Perfect prediction: {acc}")


    U = np.eye(16)
    U = U / np.linalg.norm(U)
    trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)

    setTraininglabels = np.unique(trainingLabelBitstrings, axis=0)

#    for i in range(4):
#        print(f"Zi on {i}")
#        coeffArr = generate_CoeffArr(setTraininglabels, i)
#
#        for coeff, bstring in zip(coeffArr, setTraininglabels):
#            print(f'{bstring} :  {coeff}')
#        print("")

    initialPreds = apply_U(trainingPred, U)
    accInitial = evaluate_classifier_top_k_accuracy(initialPreds, trainingLabel, 1)
    costInitial = calculate_tanhCost(initialPreds, U, trainingLabelBitstrings)

    print('Initial accuracy: ', accInitial)
    print('Initial cost: ', costInitial)
    print("")

    perfectPred = apply_U(perfectPred, U)
    accPerfect = evaluate_classifier_top_k_accuracy(perfectPred, trainingLabel, 1)
    costPerfect = calculate_tanhCost(perfectPred, U, trainingLabelBitstrings)

    print("Perfect acc: ", accPerfect)
    print("Perfect cost: ", costPerfect)

    #trainingPred = perfectPred
    #costInitial = costPerfect
    #accInitial = accPerfect

    U_update = np.copy(U) + 1e-12*np.random.randn(*U.shape)

    start = time.perf_counter()


    A = 100
    A = [100, 10, 10, 10]
    # A = [10, 100, 100, 100] is my guess for the best results but not sure
    f0 = 0.15
    f = np.copy(f0)
    decayRate = 0.2
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
                f=f, costs=True, A=A)
        updatePreds = apply_U(trainingPred, U_update)
        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        print(f'   Accuracy: {accUpdate}')
        print(f'   Cost: ', costs)
        print("")

        accuracyList.append(accUpdate)
        costsList.append(costs)
        fList.append(f)

    end = time.perf_counter()

    print(f'Elapsed time: {end - start:0.4f} seconds')

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
