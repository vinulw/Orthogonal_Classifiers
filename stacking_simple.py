import numpy as np

def evaluate_classifier_top_k_accuracy(predictions, y_test, k):
    top_k_predictions = [
        np.argpartition(image_prediction, -k)[-k:] for image_prediction in predictions
    ]
    results = np.mean([int(i in j) for i, j in zip(y_test, top_k_predictions)])
    return results

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



