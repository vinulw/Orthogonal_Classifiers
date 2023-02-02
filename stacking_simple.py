import numpy as np
from ncon import ncon

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


if __name__=="__main__":
    '''
    Use data from Lewis' dropbox
    '''
    prefix = "data_dropbox/mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    trainingPred = np.load(prefix + trainingPredPath)[15]
    trainingLabel = np.load(prefix + trainingLabelPath)
    print(trainingPred.shape)
    print(trainingLabel.shape)

    acc = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)
    print(acc)



