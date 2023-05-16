import numpy as np
import re
import os
from scipy.optimize import minimize

from stacking_simple import generate_CoeffArr, load_data, labelsToBitstrings


def tanhCost(Z, A, labelBitstrings, N=1000, labelNo=4):
    '''
    Calculate the tanh cost function using the expectation values on the label
    space `Zi` and the label space coefficients A
    '''

    totalCost = 0.0

    for i in range(labelNo):
        A_curr = A[i]
        Zi = Z[i]
        coeffArr = generate_CoeffArr(labelBitstrings, i)


        currCost = np.tanh(A_curr * Zi)
        currCost = np.einsum('i,i', coeffArr, currCost) / (N)

        totalCost += currCost
    return totalCost

def MSECost(x, Zs, accuracies, labelBs):
    A = x[:4]
    λ = x[4]

    tCosts = []
    for Z in Zs:
        tCosts.append(tanhCost(Z, A, labelBs))

    return np.linalg.norm(tCosts - λ*accuracies)


if __name__=="__main__":
    print('Loading data...')
    prefix = "data_dropbox/fashion_mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    N = 1000

    _, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

    labelBs = labelsToBitstrings(trainingLabel, 4)

    A = [100, 100, 100, 100]

    # Load some Zi

    base_path = '/mnt/c/Users/vwimalaweera/OneDrive - University College London/Project_Files/project-orthogonal_classifier/myriad_data/03052023122620_646288'

    base_path = '/mnt/c/Users/vwimalaweera/OneDrive - University College London/Project_Files/project-orthogonal_classifier/myriad_data/03052023122620_646287'

    base_path = '/mnt/c/Users/vwimalaweera/OneDrive - University College London/Project_Files/project-orthogonal_classifier/myriad_data/03052023162859_650236'

    overlap_fns = os.listdir(base_path + '/Zoverlaps')
    # Extract the steps in order
    r = re.compile('step_(\d{1,3}).npy')
    steps = []
    for fn in overlap_fns:
        searchObj = r.search(fn)
        steps.append(searchObj.group(1))

    steps = np.array(steps, dtype=float)

    # Load Zs
    overlap_fns = [base_path + '/Zoverlaps/' + fn for fn in overlap_fns]
    Zs = [np.load(fn) for fn in overlap_fns]

    # Load accuracies
    accuracy_fn = base_path + '/U_accuracy.csv'
    accuracies = np.loadtxt(accuracy_fn, delimiter=',', skiprows=1)
    accuracies = accuracies[:, 2:]
    # Order to match Zs
    argorder = [np.argwhere(accuracies[:, 0] == i).flatten()[0] for i in steps]
    accuracies = accuracies[argorder, 1]

    # Optimize MSE
    print('Minimising MSE')
    x0 = [100, 100, 100, 100, 1]

    res = minimize(MSECost, x0, args=(Zs, accuracies, labelBs))
    print(res)






