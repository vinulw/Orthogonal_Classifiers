import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import time
from ncon import ncon
from tqdm import tqdm

from stacking_simple import load_data, calculate_ZOverlap, generate_Zi, generate_CoeffArr
from stacking_simple import get_Polar

@jit
def calculate_ZOverlapJ(ϕ, U, Zi):
    return jnp.einsum('j,jk,lk,lm,m', jnp.conj(ϕ),
                      jnp.conj(U), Zi, jnp.conj(U), ϕ)

calculate_ZOverlap_batched = vmap(calculate_ZOverlapJ,
                            in_axes=(0, None, None))

@jit
def calculate_dOdVJ(ϕs, U, Zi):
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

    dOdVs = jnp.einsum('j,kl,lm,m->kj',
                      jnp.conj(ϕs), Zi, U, ϕs)

    return dOdVs

calculate_dOdV_batched = vmap(calculate_dOdVJ, in_axes=(0, None, None))

def calculate_tanhCost(ϕs, U, labelBitstrings, Zis):
    N = ϕs.shape[0]

    totalCost = 0.0

    for i, Zi in enumerate(Zis):
        coeffArr = generate_CoeffArr(labelBitstrings, i)

        Zoverlaps = jnp.real(calculate_ZOverlap_batched(ϕs, U, Zi))

        currCost = jnp.tanh(Zoverlaps)
        currCost = jnp.einsum('i,i', coeffArr, currCost) / (2*N)

        totalCost += currCost
    return totalCost

def update_U_jax(ϕs, U, labelBitstrings, Zis, f=0.1, costs=False, A=100):
    N = ϕs.shape[0]

    dZ = np.zeros(U.shape, dtype=np.complex64)
    totalCost = 0.0
    for i, Zi in enumerate(Zis):
        #print(f"On Zi : {i}")
        coeffArr = generate_CoeffArr(np.array(labelBitstrings), i)
        coeffArr = np.array(coeffArr)

        Zoverlaps = np.array((calculate_ZOverlap_batched(ϕs, U, Zi)))
        Zoverlaps = np.real(Zoverlaps)
        dOdVs = np.array(calculate_dOdV_batched(ϕs, U, Zi))

        dZi = A * (np.tanh(A)*np.cosh(A*Zoverlaps))**(-2)
        dZi = np.einsum('i,i,ijk->jk', coeffArr, dZi, dOdVs)

        dZ += dZi

        if costs:
            currCost = np.tanh(Zoverlaps)
            currCost = np.einsum('i,i', coeffArr, currCost) / (2*N)
            totalCost += currCost

    # Normalisation leads to instability
    dZ = dZ / (np.sqrt(ncon([dZ, dZ.conj()], [[1, 2], [1, 2]])) + 1e-14)

    U_update = get_Polar(np.array(U + f*dZ))

    if costs:
        return jnp.array(U_update), totalCost
    return jnp.array(U_update)


if __name__=="__main__":
    np.random.seed(1)
    N = 1000
    prefix = "data_dropbox/mnist/"
    trainingPredPath = "new_ortho_d_final_vs_training_predictions.npy"
    trainingLabelPath = "ortho_d_final_vs_training_predictions_labels.npy"

    trainingPred, trainingLabel = load_data(prefix + trainingPredPath,
              prefix + trainingLabelPath,
              N)

    print(trainingPred.shape)
    print(trainingLabel.shape)

    Zis = []
    Nq = 4

    U = np.eye(16)

    for i in range(4):
        Zis.append(generate_Zi(Nq, i+1))

    Zoverlaps = []
    tstart = time.perf_counter()
    for Zi in Zis:
        Zov = calculate_ZOverlap(trainingPred, U, Zi)
        Zoverlaps.append(Zov)
    tend = time.perf_counter()
    print(Zoverlaps[0].shape)
    print(f"Took {tend - tstart:0.4f}s to calculate overlap")

    Z0np = Zoverlaps[0].astype(np.complex64)

    trainingPredJ = jnp.array(trainingPred)
    UJ = jnp.array(U)
    Zis = [jnp.array(Zi) for Zi in Zis]

    Zoverlaps = []
    tstart = time.perf_counter()
    for Zi in Zis:
        ZovJ = calculate_ZOverlap_batched(trainingPred, UJ, Zi)
        Zoverlaps.append(ZovJ)
    tend = time.perf_counter()
    print(Zoverlaps[0].shape)
    print(f"Took {tend - tstart:0.4f}s to calculate overlap")

    Z0jax = np.array(Zoverlaps[0])

    print(np.allclose(Z0np, Z0jax))
    print("Numpy output")
    print(Z0np.shape)
    print(Z0np.dtype)
    print("Jax output")
    print(Z0jax.shape)
    print(Z0jax.dtype)

    from stacking_simple import labelsToBitstrings, apply_U
    from stacking_simple import evaluate_classifier_top_k_accuracy
    import matplotlib.pyplot as plt
    trainingLabelBitstrings = labelsToBitstrings(trainingLabel, 4)
    trainingLabelBitstrings = jnp.array(trainingLabelBitstrings)

    f0 = 0.2
    decayRate = 0.05
    def curr_f(decayRate, itNumber, initialRate):
        return initialRate / (1 + decayRate * itNumber)

    U_update = np.copy(U) + 1e-3*np.random.randn(*U.shape)
    U_update = jnp.array(U_update)
    Nsteps = 250

    costInit = calculate_tanhCost(trainingPredJ, U_update, trainingLabelBitstrings, Zis)
    trainingPred = apply_U(trainingPred, np.array(U_update))
    accInit = evaluate_classifier_top_k_accuracy(trainingPred, trainingLabel, 1)
    costsList = [costInit]
    accuracyList = [accInit]
    fList = []

    print(f"Initial Accuracy: {accInit}")
    print(f"Initial Costs: {costInit}")

    for i in tqdm(range(Nsteps)):
        #print(f'Update step {i+1}')
        f = curr_f(decayRate, i, f0)
        if f < 5e-4:
            f = 5e-4
        #print(f'   f: {f}')
        U_update, costs = update_U_jax(trainingPredJ, U_update, trainingLabelBitstrings,
                                       f=f, costs=True, Zis=Zis)
        trainingPred = np.array(trainingPredJ)
        updatePreds = apply_U(trainingPred, np.array(U_update))
        accUpdate = evaluate_classifier_top_k_accuracy(updatePreds, trainingLabel, 1)
        #print(f'   Accuracy: {accUpdate}')
        #print(f'   Cost: ', costs)
        #print("")

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
