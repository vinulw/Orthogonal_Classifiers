from stacking_simple import load_data
from stacking_simple import evaluate_classifier_top_k_accuracy, labelsToBitstrings
from stacking_simple import apply_U_rho, trace_rho, get_Polar
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__=="__main__":
    np.random.seed(1)
    N = 1000
    prefix = "data_dropbox/fashion_mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"
    trainingPred, trainingLabel = load_data(prefix + trainingPredPath, prefix
            + trainingLabelPath, N)

    trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)

    acc = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)
    print('Initial accuracy: ', acc)

    trainingPred = np.array([np.kron(im, im) for im in trainingPred])
    ρPred = np.array([np.outer(pred, pred.conj()) for pred in trainingPred])
    qNo = 8

    classfier_path = 'tanh_train/03042023110838/classifier_U/'
    fnames = os.listdir(classfier_path)

    accuracies = np.zeros(len(fnames))
    accuracies_Polar = np.zeros(len(fnames))
    steps = np.zeros(len(fnames))
    for i, fname in tqdm(enumerate(fnames), total=len(fnames)):
        step = int(fname[5:-4])
        U = np.load(classfier_path + fname, allow_pickle=True)
        U_polar = get_Polar(U)

        updaterho = apply_U_rho(ρPred, U)
        updatePreds = trace_rho(updaterho, qNo, trace_ind=[0, 1, 2, 3])
        updatePreds = np.diagonal(updatePreds, axis1=1, axis2=2) # Get Tr(Pi ρ)
        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)

        updaterho_P = apply_U_rho(ρPred, U_polar)
        updatePreds_P = trace_rho(updaterho_P, qNo, trace_ind=[0, 1, 2, 3])
        updatePreds_P = np.diagonal(updatePreds_P, axis1=1, axis2=2) # Get Tr(Pi ρ)
        accUpdate_P = evaluate_classifier_top_k_accuracy(updatePreds_P, trainingLabel, 1)

        accuracies[i] = accUpdate
        accuracies_Polar[i] = accUpdate_P
        steps[i] = step

    sortind = np.argsort(steps)
    steps =  steps[sortind]
    accuracies, accuracies_Polar = accuracies[sortind], accuracies_Polar[sortind]
    plt.plot(steps, accuracies, '.--', label='Non-orthogonal')
    plt.plot(steps, accuracies_Polar, '.--', label='Orthogonal')
    plt.show()

