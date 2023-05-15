import numpy as np
import os
from stacking_simple import apply_U_rho, load_data, labelsToBitstrings, evaluate_classifier_top_k_accuracy
from stacking_three import copy_state, pred_U_state
from stacking_simple import calculate_ZOverlap, generate_Zi
import re
from tqdm import tqdm
import sys

def get_meta_info(path):
    '''
    Get meta information about classifier including the data, job id (if present) and step number.

    Paths should be of the form:
    - `.../{date}_{job_id}/.../step_{step_no}.npy`
    - `.../{date}/.../step_{step_no}.npy`

    Where `date` is a 14 digit timestamp. `job_id` is 6 digits and `step_no` is number between (10, 999)
    '''
    r_date_jobid = re.compile('(\d{14})_?(\d{6})?')
    r_step = re.compile('step_(\d{1,3}).npy')

    searchObj = r_date_jobid.search(path)
    date = searchObj.group(1)
    jobid = searchObj.group(2)

    searchObj = r_step.search(path)
    step = searchObj.group(1)

    return date, jobid, step

if __name__=="__main__":

    base_path = '/mnt/c/Users/vwimalaweera/OneDrive - University College London/Project_Files/project-orthogonal_classifier/myriad_data/03052023162859_650236'

    path = base_path + '/classifier_U'

    U_paths = [path + '/' + p for p in os.listdir(path)]

    print('Loading data...')
    prefix = "data_dropbox/fashion_mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    N = 1000

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

    for p in tqdm(U_paths, total=len(U_paths)):
        date, jobid, step = get_meta_info(p)
        tqdm.write(f'Working on {date}_{jobid}_{step}')
        U = np.load(p)
        n_copies = int(np.emath.logn(16, U.shape[0]))

        trainStates = np.array([copy_state(s, n_copies) for s in trainingPred])

        tqdm.write('   Calculating accuracy...')
        # Save accuracy
        preds = pred_U_state(trainStates, U)
        acc = evaluate_classifier_top_k_accuracy(preds, trainingLabel, 1)

        accFile = base_path + '/U_accuracy.csv'

        if not os.path.exists(accFile):
            with open(accFile, 'w') as f:
                line = 'Date, JobId, Step, Accuracy\n'
                f.write(line)

        tqdm.write('   Saving accuracy...')
        with open(accFile, 'a') as f:
            line = f'{date}, {jobid}, {step}, {acc}\n'
            f.write(line)

        tqdm.write('   Calculating Z Overlaps...')
        # Calculate Z Overlaps
        qNo = int(np.log2(trainStates.shape[1]))
        label_start = qNo-4

        Zls = np.zeros((4, trainStates.shape[0]))

        for i in range(qNo - label_start):
            Zi = generate_Zi(qNo, i+1+label_start)
            Zoverlaps = np.real(calculate_ZOverlap(trainStates, U, Zi))
            Zls[i] = Zoverlaps

        tqdm.write('   Saving Z Overlaps...')
        Zdir = base_path + '/Zoverlaps'

        if not os.path.exists(Zdir):
            os.makedirs(Zdir)

        Zfn = Zdir + '/' + f'step_{step}.npy'

        np.save(Zfn, Zls)


