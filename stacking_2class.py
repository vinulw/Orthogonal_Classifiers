import numpy as np
from ncon import ncon


Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

def getLabelSpace(data, U):
    '''
    Apply a feature extractor U to the data to get the label space vector.
    '''
    return ncon([U, data], [[-2, 1], [-1, 1]])

def getNCopiedSpace(F, Ncopies):
    assert Ncopies > 1

    i = 97
    constr = '{}{},'.format(chr(i), chr(i+1))
    endstr = '{}{}'.format(chr(i), chr(i+1))
    curri = i + 2
    for j in range(Ncopies-1):
        constr += '{}{},'.format(chr(i), chr(curri))
        endstr += '{}'.format(chr(curri))
        curri += 1

    constr = constr[:-1] + '->' + endstr
    FN =  np.einsum(constr, *[F]*Ncopies)

    N, _ = F.shape
    return FN.reshape(N, -1)

def partialZMeasureOld(states, Ncopies, V):
    ZN = ncon([Z, np.eye(2**(Ncopies-1))], ((-1, -3), (-2, -4)))
    ZN = ZN.reshape(2**Ncopies, 2**Ncopies)

    measures = np.zeros(N)
    for i in range(N):
        di = states[i,:]
        measurei = np.real(ncon([np.conj(di), np.conj(V), ZN, V, di]
                    , ([[1], [2, 1], [3, 2], [3, 4], [4]])))
        measures[i] = measurei

    return measures

def overlapRemovedVOld(states, Ncopies, V):
    ZN = ncon([Z, np.eye(2**(Ncopies-1))], ((-1, -3), (-2, -4)))
    ZN = ZN.reshape(2**Ncopies, 2**Ncopies)

    D = 2**Ncopies

    dZs = np.zeros((N, D, D), dtype=complex)
    for i in range(N):
        di = states[i,:]
        dZ = ncon([np.conj(di), ZN, V, di]
                    , ([[-2], [-1, 1], [1, 2], [2]]))
        dZs[i] = dZ

    return dZs


def overlapRemovedV(states, Ncopies, V):
    ZN = ncon([Z, np.eye(2**(Ncopies-1))], ((-1, -3), (-2, -4)))
    ZN = ZN.reshape(2**Ncopies, 2**Ncopies)

    dZs = np.einsum('ij,kl,lm,im->ikj',
                    np.conj(states), ZN, V, states)
    return dZs


def partialZMeasure(states, Ncopies, V):
    ZIN = ncon([Z, np.eye(2**(Ncopies-1))], ((-1, -3), (-2, -4)))
    ZIN = ZIN.reshape(2**Ncopies, 2**Ncopies)

    measures = np.einsum('ij,kj,lk,lm,im->i',
            np.conj(states), np.conj(V), ZIN, V, states)

    return measures

def get_dZandC(states, Ncopies, V, A=10):
    overlapZ = np.real(partialZMeasure(states, Ncopies, V))
    doverlapZ_V = overlapRemovedV(states, Ncopies, V)

    c = A / (np.tanh(A)*(np.cosh(A*np.sign(overlapZ)*np.absolute(overlapZ)))**2)
    dZ = ncon([c, doverlapZ_V], [[1,], [1, -1, -2]])

    return dZ, np.sum(c)

def optimiseVPolar(states1, states2, Ncopies, V, A=10, f=50, m=50):
    m = 0
    cost, accuracy = getCostandAccuracy(states1, states2, Ncopies, V, A)
    costs = [cost]
    accuracies = [accuracy]
    while m < 50:
        dZ1, c1 = get_dZandC(states1, Ncopies, V, A)
        dZ2, c2 = get_dZandC(states2, Ncopies, V, A)

        C = c1 - c2

        dZ = dZ1 - dZ2
        dZ = dZ / np.sqrt(ncon([dZ, dZ.conj()], [[1, 2], [1, 2]]))

        V = get_Polar(dZ + f*V)
        m += 1
        cost, accuracy = getCostandAccuracy(states1, states2, Ncopies, V, A)
        costs.append(cost)
        accuracies.append(accuracy)
    return V, costs, accuracies

def getCostandAccuracy(states1, states2, Ncopies, V, A=10):
    overlapZ1 = np.real(partialZMeasure(states1, Ncopies, V))

    overlapZ2 = np.real(partialZMeasure(states2, Ncopies, V))

    N = states1.shape[0]

    Cost = (np.sum(np.tanh(overlapZ1)) - np.sum(np.tanh(overlapZ2)))/(2*N)
#    print(overlapZ1)
#    print()
#    print(overlapZ2)

    acc1 = 1.0 + np.sign(overlapZ1)
#    print(acc1)
#    print()
    acc1 = np.sum(acc1)
    acc2 = 1.0 - np.sign(overlapZ2)
#    print(acc2)
    acc2 = np.sum(acc2)

    Accuracy = (acc1 + acc2)/(4*N)

    return Cost, Accuracy


def get_Data(N,Theta,phi,chi,dx,dy):
    """ Returns a set of spinors on the block shere """
    """ i. centred at theta,phi """
    """ ii. from sterographic projection of gaussian variance dx,dy"""
    """ iii. rotated to an angle chi in plane"""
    X = np.random.normal(0.0,dx,N)
    Y = np.random.normal(0.0,dy,N)
    Thet = 2*np.arctan2((np.sqrt(X*X+Y*Y)),2)
    Ph = np.arctan2(X,Y)+chi
    Z = np.zeros((N,2),dtype=complex)
    m = 1
    while m < N+1:
        Z[m-1,0] =  np.cos(Thet[m-1]/2)+1j*0.0
        Z[m-1,1] = np.sin(Thet[m-1]/2)*np.exp(1j*Ph[m-1])
        m +=1
    R = np.zeros((2,2),dtype=complex)
    R[0,0] = np.cos(Theta/2)
    R[0,1] = np.sin(Theta/2)*np.exp(1j*phi)
    R[1,0] = -np.sin(Theta/2)*np.exp(-1j*phi)
    R[1,1] = np.cos(Theta/2)
    Z = ncon([R,Z],([-2,1],[-1,1]))
    return Z

def get_Features(D1,D2):
    """ construct the classifier initialisation from Z1 and Z2 at order n"""
    s1,s2 = get_s(2)
    aveD1 = np.sum(D1,axis=0)
    aveD1 = aveD1/(np.sqrt(ncon([aveD1,np.conj(aveD1)],([1],[1]))))
    aveD2 = np.sum(D2,axis=0)
    aveD2 = aveD2/(np.sqrt(ncon([aveD2,np.conj(aveD2)],([1],[1]))))
    U = ncon([s1,np.conj(aveD1)],([-1],[-2]))\
    +ncon([s2,np.conj(aveD2)],([-1],[-2]))
    U = get_Polar(U)
    return U

def get_s(d):
    """ returns a spin up vector useful later"""
    s1=np.zeros((d),dtype=np.float64)
    s1[0]=1.0
    s2=np.zeros((d),dtype=np.float64)
    s2[1]=1.0
    return s1,s2

def get_Polar(M):
    """ Return the polar decomposition of M """
    x,y,z =  np.linalg.svd(M)
    M = ncon([x,z],([-1,1],[1,-2]))
    return M

def get_Classify1(N,U,D1,D2):
    A=5
    Z = np.eye(2)
    Z[1,1] = - Z[1,1]
    """ prob of right for 1 minus prob of wrong """
    c1 = np.real(ncon([np.conj(D1),np.conj(U),Z,U,D1]\
    ,([5,1],[2,1],[2,3],[3,4],[5,4])))/(2*N)
    """ prob of right for 2 minus prob of wrong """
    c2 = np.real(ncon([np.conj(D2),np.conj(U),Z,U,D2]\
    ,([5,1],[2,1],[2,3],[3,4],[5,4])))/(2*N)
    Cost = c1-c2
    """ An alternative effectively taking contribution of each image ^1/3 """
    l=1
    ccc1 = 0.0
    ccc2 = 0.0
    ov1 = np.zeros(N)
    while l < N+1:
        D1temp = D1[l-1,:]
        D2temp = D2[l-1,:]
        c1temp = np.real(ncon([np.conj(D1temp),np.conj(U),Z,U,D1temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))
        ov1[l-1] = c1temp
        c2temp = np.real(ncon([np.conj(D2temp),np.conj(U),Z,U,D2temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))
        """ccc1 = ccc1 + np.sign(c1temp)*(np.absolute(c1temp))**(1/3)"""
        """ccc2 = ccc2 + np.sign(c2temp)*(np.absolute(c2temp))**(1/3)"""
        ccc1 = ccc1 + np.tanh(A*np.sign(c1temp)*(np.absolute(c1temp)))/np.tanh(A)
        ccc2 = ccc2 + np.tanh(np.sign(c2temp)*(np.absolute(c2temp)))/np.tanh(A)
        l+=1

    myOv1 = np.real(partialZMeasure(D1, 1, U))
    print("Checking overlaps are close...")
    print(np.allclose(myOv1, ov1))
    print('Checking sum of overlaps close...')
    myOv1 = np.sum(myOv1)/(2*N)
    print(np.allclose(np.sum(myOv1), c1))
    Cost2 = (ccc1-ccc2)/N
    """ Now calculate an equivalent of the training accuracy """
    l=1
    acc1 = 0.0
    acc2 = 0.0
    acc1s = np.zeros(N)
    acc2s = np.zeros(N)
    while l < N+1:
        D1temp = D1[l-1,:]
        D2temp = D2[l-1,:]
        acc1_ = (1.0+np.sign(np.real(ncon([np.conj(D1temp),
                np.conj(U),Z,U,D1temp] ,([1],[2,1],[3,2],[3,4],[4])))))
        acc1s[l-1] = acc1_
        acc1 = acc1 + acc1_
        acc2_ = (1.0-np.sign(np.real(ncon([np.conj(D2temp),
            np.conj(U),Z,U,D2temp] ,([1],[2,1],[3,2],[3,4],[4])))))
        acc2s[l-1] = acc2_
        acc2 = acc2 + acc2_
        l+=1
    overlapZ1 = np.real(partialZMeasure(D1, 1, U))
    myAcc1 = 1.0 + np.sign(overlapZ1)
    overlapZ2 = np.real(partialZMeasure(D2, 1, U))
    myAcc2 = 1.0 - np.sign(overlapZ2)
    print('Checking if acc matches...')
    print(np.allclose(acc1s, myAcc1))
    print(np.allclose(acc2s, myAcc2))
    print('Acc1')
    print(np.sum(myAcc1))
    print(acc1)
    print('Acc2')
    print(np.sum(myAcc2))
    print(acc2)
    myAcc1 = np.sum(myAcc1)
    myAcc2 = np.sum(myAcc2)
    print((myAcc1 + myAcc2)/(4*N))
    Accuracy = (acc1+acc2)/(4*N)
    return Cost,Cost2,Accuracy

if __name__ =="__main__":
    np.random.seed(1)
    N = 1000
#    N = 10
    Theta1 = -1
    phi1 = 0.0
    chi1 = 0.6
    dx1 = 0.4
    dy1 = 0.8
    Theta2 = 0
    phi2 = 0.0
    chi2 = 0.0
    dx2 = 0.025
    dy2 = 0.025

    D1 = get_Data(N,Theta1,phi1,chi1,dx1,dy1)
    D2 = get_Data(N,Theta2,phi2,chi2,dx2,dy2)

    U = get_Features(D1,D2)

    F1 = getLabelSpace(D1, U)
    F2 = getLabelSpace(D2, U)


#    U2 = ncon([U, np.eye(2)], [[-1, -3], [-2, -4]]).reshape(2**2, 2**2)

#    dZ, c = get_dZandC(F, 1, U)

    V1, costs, accuracies = optimiseVPolar(D1, D2, 1, U)
    print(V1.shape)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(costs)
    plt.title('Costs')

    plt.figure()
    plt.plot(accuracies)
    plt.title('Accuracies')
    plt.show()

#    V2 = ncon([V1, np.eye(2)], [[-1, -3], [-2, -4]]).reshape(2**2, 2**2)
#    FF1 = getNCopiedSpace(F1, 2)
#    FF2 = getNCopiedSpace(F2, 2)
#
#    V2 = optimiseVPolar(FF1, FF2, 2, np.eye(4))
#    print(V2.shape)
#
#    print(getCostandAccuracy(D1, D2, 1, U), ' Before optimisation 1 copy')
##    print(get_Classify1(N, U, D1, D2), ' Andrew Cost Accuracy')
#    print(getCostandAccuracy(D1, D2, 1, V1), ' After optimisation 1 copy')
#    print(getCostandAccuracy(FF1, FF2, 2, V2), ' After optimisation 2 copy')
#
#    V3 = ncon([V2, np.eye(2)], ((-1, -3), (-2, -4))).reshape([2**3, 2**3])
#    FFF1 = getNCopiedSpace(F1, 3)
#    FFF2 = getNCopiedSpace(F2, 3)
#
#    print(getCostandAccuracy(FFF1, FFF2, 3, V3), ' Before optimisation 3 copy')
#    V3 = optimiseVPolar(FFF1, FFF2, 3, V3)
#
#    print(getCostandAccuracy(FFF1, FFF2, 3, V3), ' After optimisation 3 copy')
