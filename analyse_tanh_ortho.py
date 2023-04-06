from stacking_simple import load_data
from stacking_simple import evaluate_classifier_top_k_accuracy, labelsToBitstrings
from stacking_simple import apply_U_rho, trace_rho, get_Polar
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_train_data(csv_path):
    with open(csv_path, 'r') as f:
        data = np.loadtxt(f, delimiter=',')
    accuracies, costs, lrs = data.astype(float).T
    plt.figure()
    plt.plot(accuracies)
    plt.title('Accuracies')

    plt.figure()
    plt.plot(costs)
    plt.title('Costs')

    plt.figure()
    plt.plot(lrs)
    plt.title('Learning Rates')

    plt.show()

def plot_schmidt_Us(U_path):
    import time
    from svd_robust import svd
    from matplotlib.widgets import Slider
    from tqdm import tqdm

    fles = os.listdir(U_path)
    print('Loading Us : ', U_path)

    steps = [None] * len(fles)
    Us = [None] * len(fles)
    for i, f in enumerate(fles):
        steps[i] = int(f[5:-4])
        Us[i] = np.load(os.path.join(U_path, f), allow_pickle=True)

    steps, Us = zip(*sorted(zip(steps, Us), key=lambda x: x[0]))

    U = Us[-1] # Get the last U

    # Look at the Schmidt spectrum
    schmidts = [None] * len(fles)
    maxLen = Us[0].shape[0]
    for i, U in tqdm(enumerate(Us), total=len(Us)):
        _, S, _ = svd(U)
        #lenS = len(S)
        #schmidts[i] = np.pad(S, (0, maxLen-lenS), constant_value=(0, 0))
        schmidts[i] = S

    xs = range(maxLen)

    fig, ax = plt.subplots()
    ax.set_ylim(0.0, 0.4)
    line, = ax.plot(xs, schmidts[0], '--')
    txt = ax.annotate(f'Step: {steps[0]}', (0., 0.35))

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)
    # Make a horizontal slider to control the frequency.
    axstep = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    step_slider = Slider(
        ax=axstep,
        label='Step Index',
        valmin=0,
        valmax=len(schmidts)-1,
        valstep=1.,
        valinit=0,
    )
    # The function to be called anytime a slider's value changes
    def update(val):
        i = int(step_slider.val)
        line.set_ydata(schmidts[i])
        txt.set_text(f'Step: {steps[i]}')
        fig.canvas.draw_idle()


    # register the update function with each slider
    step_slider.on_changed(update)
    plt.show()



if __name__=="__main__":
    #plot_schmidt_Us('tanh_train/03042023120628/classifier_U/')
    #path = 'tanh_train/04042023182002/run_data.csv'
    #plot_train_data(path)

    from svd_robust import svd
    from ncon import ncon
    np.set_printoptions(precision=3)
    U_path = 'tanh_train/03042023120628/classifier_U/'

    fles = os.listdir(U_path)
    print('Loading Us : ', U_path)

    steps = [None] * len(fles)
    Us = [None] * len(fles)
    for i, f in enumerate(fles):
        steps[i] = int(f[5:-4])
        Us[i] = np.load(os.path.join(U_path, f), allow_pickle=True)

    steps, Us = zip(*sorted(zip(steps, Us), key=lambda x: x[0]))

    U = Us[-1] # Get the last U

    U_trunc, S, V = svd(U)

    U_trunc = U_trunc[:, :16]
    S_trunc = S[:16]
    print('Singualr values sum: ', np.sum(S))
    print('Truncated singualr values sum: ', np.sum(S_trunc))
    assert()

    V_trunc = V[:16, :]

#    V0 = V[0]
#    V1 = V[1]
#
#    V0 = V0.reshape(16, 16)
#    V1 = V1.reshape(16, 16)
#
#    V0V1 = ncon([V0, V1.conj()], ((1, -1), (1, -2)))
#    V0V0 = ncon([V0, V0.conj()], ((1, -1), (1, -2)))
#    V1V1 = ncon([V1, V1.conj()], ((1, -1), (1, -2)))
#
#    # Truncation
#    V0V1[V0V1 < 9e-3] = 0
#    V0V0[V0V0 < 9e-3] = 0
#    V1V1[V1V1 < 9e-3] = 0
#
#    print('V0V1')
#    print(V0V1)
#    print()
#    print('V0V0')
#    print(V0V0)
#    print('V1V1')
#    print(V1V1)

    #U_trunc = U_trunc @ np.diag(S_trunc) @ V_trunc
    U_trunc = U_trunc @  V_trunc

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

#    updaterho = apply_U_rho(ρPred, U_trunc)
#    updatePreds = trace_rho(updaterho, qNo, trace_ind=[0, 1, 2, 3])
#    updatePreds = np.diagonal(updatePreds, axis1=1, axis2=2) # Get Tr(Pi ρ)
#    accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
#
#    print('Truncated U accurcy: ', accUpdate)

    U_polar = get_Polar(U)
    updaterho = apply_U_rho(ρPred, U_polar)
    updatePreds = trace_rho(updaterho, qNo, trace_ind=[0, 1, 2, 3])
    updatePreds = np.diagonal(updatePreds, axis1=1, axis2=2) # Get Tr(Pi ρ)
    accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
    print('U polar accurcy: ', accUpdate)

    n_step = 20
    U = Us[-1] # Get the last U

    def dV(M, V):
        M_dag = M.conj().T
        M = M.reshape(16, 16, 16, 16)
        M_dag = M_dag.reshape(16, 16, 16, 16)
        V = V.reshape(16, 16, 16, 16)
        dV = ncon([M, M_dag, V], ((-1, -2, 1, -4), (1, 2, 3, 4), (3, 4, -3, 2)))
        dV = dV.reshape(16**2, 16**2)
        return dV

    def stochastic_update_V(V, alpha=0.1):
        dim = V.shape
        V_update = V + alpha*(np.random.rand(dim), + 1j*np.random.rand(dim))
        V_update = get_Polar(V)
        return V_update

    def cost(V, M):
        V_dag = V.conj().T
        M_dag = M.conj().T

        M = M.reshape(16, 16, 16, 16)
        V = V.reshape(16, 16, 16, 16)
        M_dag = M_dag.reshape(16, 16, 16, 16)
        V_dag = V_dag.reshape(16, 16, 16, 16)




    V = np.copy(U_trunc) # Initial V
    α = -0.5
    updaterho = apply_U_rho(ρPred, V)
    updatePreds = trace_rho(updaterho, qNo, trace_ind=[0, 1, 2, 3])
    updatePreds = np.diagonal(updatePreds, axis1=1, axis2=2) # Get Tr(Pi ρ)
    accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
    print('V initial accurcy: ', accUpdate)

    updaterho = apply_U_rho(ρPred, U)
    updatePreds = trace_rho(updaterho, qNo, trace_ind=[0, 1, 2, 3])
    updatePreds = np.diagonal(updatePreds, axis1=1, axis2=2) # Get Tr(Pi ρ)
    accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
    print('Target accurcy: ', accUpdate)
    for i in range(n_step):
        print('Current step: ', i)
        dV_curr = dV(U, V)
        V = get_Polar(V + α*dV_curr)

        updaterho = apply_U_rho(ρPred, V)
        updatePreds = trace_rho(updaterho, qNo, trace_ind=[0, 1, 2, 3])
        updatePreds = np.diagonal(updatePreds, axis1=1, axis2=2) # Get Tr(Pi ρ)
        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        print('V accurcy: ', accUpdate)
    assert()

    # classfier_path = 'tanh_train/03042023110838/classifier_U/'
    classfier_path = 'tanh_train/03042023120628/classifier_U/'
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
    plt.plot(steps, accuracies, '.--', label='Non Orthogonal')
    plt.plot(steps, accuracies_Polar, '.--', label='Orthogonal')
    plt.legend()

    # Save the output
    save_path = 'tanh_train/03042023120628/ortho_acc_data.csv'
    header = 'step, accuracies_non_orth, accuracies_orth'
    data = np.array([steps, accuracies, accuracies_Polar])
    data = data.T
    np.savetxt(save_path, data, delimiter=',', header=header)
    plt.show()


