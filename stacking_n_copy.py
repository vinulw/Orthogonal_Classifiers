import numpy as np
from ncon import ncon
from math import floor
import matplotlib.pyplot as plt
from collections.abc import Iterable
from opt_einsum import contract

import os
from datetime import datetime
import time
import json

def evaluate_classifier_top_k_accuracy(predictions, y_test, k):
    top_k_predictions = [
        np.argpartition(image_prediction, -k)[-k:] for image_prediction in predictions
    ]
    results = np.mean([int(i in j) for i, j in zip(y_test, top_k_predictions)])
    return results

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
        assert(np.all(labels[labelArgs] == i))

    return outStates, outlabels

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

    for i in range(qNo - label_start):
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

    #Zmeasure = np.einsum('ij,kj,lk,lm,im->i',
    #                     np.conj(ϕs), np.conj(U), Zi, U, ϕs)
    Zmeasure = contract('ij,kj,lk,lm,im->i',
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

    dOdVs = contract('ij,kl,lm,im->ikj',
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

    return contract('lm,im->il', U, ϕs)

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
    print('Iterated qNo: ', qNo - label_start)
    for i in range(qNo - label_start):
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
        dZi = contract('i,i,ijk->jk', coeffArr, dZi, dOdVs)

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
#    from svd_robust import svd
#    x,y,z =  svd(M)
#    M = ncon([x,z],([-1,1],[1,-2]))
    from scipy.linalg import polar
    M, _ = polar(M)
    return M

def plot_qubit_histogram(predVecs, trainingLabelBitstrings, title, show=True, save_name=None):
    qNo = int(np.log2(predVecs.shape[1]))
    U = np.eye(2**qNo)

    fig, axs = plt.subplots(1, qNo, figsize=(20, 5))

    for i in range(qNo):
        Zi = generate_Zi(qNo, i+1)

        ZiOverlaps = np.real(calculate_ZOverlap(predVecs, U, Zi))
        coeffArr = generate_CoeffArr(trainingLabelBitstrings, i)

        Zi_minus = ZiOverlaps[coeffArr == -1]
        Zi_plus = ZiOverlaps[coeffArr == 1]

        ax = axs[i]
        ax.hist(Zi_minus, alpha=0.5)
        ax.hist(Zi_plus, color='r', alpha=0.5)
        ax.set_title(f'i: {i} ')

    plt.tight_layout(pad=2.0, w_pad=5., h_pad=10.0)
    fig.suptitle(title)
    if save_name is not None:
        plt.savefig(save_name)
    if show:
        plt.show()
    plt.close()

def train_N_copy(config_path):
    now = datetime.now()
    now = now.strftime('%d%m%Y%H%M%S')

    np.random.seed(1)
    #prefix = "data_dropbox/mnist/"
    prefix = "data_dropbox/fashion_mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    save_dir = f'tanh_train/{now}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'Made save directory: {save_dir}')
    show = False

    N = 1000

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

    acc = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)
    print('Initial accuracy: ', acc)

    trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)

    with open(config_path, 'r') as cf:
        config_dict = json.load(cf)

    Ncopies = config_dict['Ncopies']
    # Make N copies
    label_start = 0
    if Ncopies == 2:
        label_start = 4
    elif Ncopies == 3:
        label_start = 8
    qNo = 4*Ncopies
    dim_N = 2**qNo

    if Ncopies == 2:
        trainingPred = np.array([np.kron(im, im) for im in trainingPred])
    if Ncopies == 3:
        trainingPred_ = np.array([np.kron(im, im) for im in trainingPred])
        trainingPred = np.array([np.kron(im, impred) for im, impred in zip(trainingPred, trainingPred_)])

    print('The sizes of objects: ')
    print('trainingPred.shape: ', trainingPred.shape)
    print('trainingPred: ', trainingPred.nbytes)
    ρPred = contract('ij, ik -> ijk', trainingPred, trainingPred.conj())
    print('ρPred.shape: ', ρPred.shape)

    U = np.eye(dim_N)
    U = U / np.linalg.norm(U)
    U_update = np.copy(U) + 1e-12*np.random.randn(*U.shape)

    print('ρPred: ', ρPred.nbytes)
    print('Uupdate: ', U_update.nbytes)

    assert()
    initialrho = apply_U_rho(ρPred, U)
    if Ncopies == 2:
        initialrho_ = trace_rho(initialrho, qNo, trace_ind=[0, 1, 2, 3])
    elif Ncopies == 3:
        initialrho_ = trace_rho(initialrho, qNo, trace_ind=[0, 1, 2, 3, 4, 5, 6, 7])
    else:
        initialrho_ = trace_rho(initialrho, qNo, trace_ind=[])
    initialPreds = np.diagonal(initialrho_, axis1=1, axis2=2) # Get Tr(Pi ρ)

    accInitial = evaluate_classifier_top_k_accuracy(initialPreds, trainingLabel, 1)
    costInitial = calculate_tanhCost(trainingPred, U, trainingLabelBitstrings, label_start=label_start)

    print('Initial accuracy: ', accInitial)
    print('Initial cost: ', costInitial)
    print("")


    start = time.perf_counter()

    # Saving classifiers
    save_interval = 10
    classifier_dir = save_dir + 'classifier_U/'
    if not os.path.exists(classifier_dir):
        os.makedirs(classifier_dir)
        print(f'Made classifier directory: {classifier_dir}')


    # Load training config
    As = config_dict["As"]
    switch_index = config_dict["switch_index"]
    Nsteps = config_dict["Nsteps"]

    Ai = 0

    f0 = config_dict["f0"]
    decayRate = config_dict["decayRate"]

    f = np.copy(f0)
    def curr_f(decayRate, itNumber, initialRate):
        return initialRate / (1 + decayRate * itNumber)

    # CSV file to track training
    csv_data_file = save_dir + 'run_data.csv'

    with open(csv_data_file, 'w') as f:
        header = 'accuracy, cost, lr'
        line = np.array([accInitial, costInitial, f0])
        np.savetxt(f, line.reshape(1, -1), delimiter=', ', header=header)


    costsList = [costInitial]
    accuracyList = [accInitial]
    fList = []
    ortho_step = Nsteps + 10
    i = 0
    tol = 0.05
    plot_qubit_histogram(initialPreds, trainingLabelBitstrings,
            'Initial histogram', show=False, save_name=save_dir + 'initial_hist.png')
    for n in range(Nsteps):
        A = As[Ai]
        print(f'Update step {n+1}')
        f = curr_f(decayRate, i, f0)
        if f < 2e-3:
            f = 2e-3
        print(f'   f: {f}')
        U_update, costs = update_U(trainingPred, U_update, trainingLabelBitstrings,
                f=f, costs=True, A=A, label_start=label_start)
        updaterho = apply_U_rho(ρPred, U_update)
        if Ncopies == 2:
            updatePreds = trace_rho(updaterho, qNo, trace_ind=[0, 1, 2, 3])
        if Ncopies == 3:
            updatePreds = trace_rho(updaterho, qNo, trace_ind=[0, 1, 2, 3, 4, 5, 6, 7])
        else:
            updatePreds = trace_rho(updaterho, qNo, trace_ind=[])
        updatePreds = np.diagonal(updatePreds, axis1=1, axis2=2) # Get Tr(Pi ρ)
        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        print(f'   Accuracy: {accUpdate}')
        print(f'   Cost: ', costs)
        print("")

        accuracyList.append(accUpdate)
        costsList.append(costs)
        fList.append(f)

        with open(csv_data_file, 'a') as fle:
            line = np.array([accUpdate, costs, f])
            np.savetxt(fle, line.reshape(1, -1), delimiter=', ')

        #if n > 10 and np.std(costsList[:-10]) <= tol and Ai < len(Ai):
        # For running with MNIST
        #if n == 30:
        #    print('Resetting Ai and f0')
        #    Ai += 1
        #    f0 = f0*0.8
        #    i = 0
        # For running with Fashion MNIST
        if n in switch_index:
            print('Resetting Ai and f0')
            Ai += 1
            f0 = f0*0.8
            i = 0

        if n % save_interval == 0:
            save_name = save_dir + f'step_{n}_hist.png'
            plot_qubit_histogram(updatePreds, trainingLabelBitstrings,
                    title=f'Step: {n}', show=False, save_name=save_name)
            classifier_name = classifier_dir + f'step_{n}.npy'
            np.save(classifier_name, U_update)

        if n % ortho_step:
            U_update = get_Polar(U_update)

        i += 1



    end = time.perf_counter()

    print(f'Elapsed time: {end - start:0.4f} seconds')

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


if __name__=="__main__":
    config_path = 'experiment_param_three.json'

    train_N_copy(config_path)


