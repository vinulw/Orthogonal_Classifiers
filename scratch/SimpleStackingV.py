from scipy import integrate
from scipy.io import loadmat
from ncon import ncon
import matplotlib.pyplot as plt
from scipy.linalg import expm
import scipy as sp
from pylab import *
import numpy as np
import scipy.sparse.linalg.eigen.arpack as arp
import pylab as pl
import tables
from scipy import integrate

def init_V(D):
    """ Returns a random unitary """
    A = np.random.rand(D,D)
    V,x = np.linalg.qr(A)
    return V

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

def get_Classify1(N,U,D1,D2):
    A=5
    s1,s2 = get_s(2)
    Z = eye(2)
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
    while l < N+1:
        D1temp = D1[l-1,:]
        D2temp = D2[l-1,:]
        c1temp = np.real(ncon([np.conj(D1temp),np.conj(U),Z,U,D1temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))
        c2temp = np.real(ncon([np.conj(D2temp),np.conj(U),Z,U,D2temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))
        """ccc1 = ccc1 + np.sign(c1temp)*(np.absolute(c1temp))**(1/3)"""
        """ccc2 = ccc2 + np.sign(c2temp)*(np.absolute(c2temp))**(1/3)"""
        ccc1 = ccc1 + np.tanh(A*np.sign(c1temp)*(np.absolute(c1temp)))/np.tanh(A)
        ccc2 = ccc2 + np.tanh(np.sign(c2temp)*(np.absolute(c2temp)))/np.tanh(A)
        l+=1
    Cost2 = (ccc1-ccc2)/N
    """ Now calculate an equivalent of the training accuracy """
    l=1
    acc1 = 0.0
    acc2 = 0.0
    while l < N+1:
        D1temp = D1[l-1,:]
        D2temp = D2[l-1,:]
        acc1 = acc1 + (1.0+np.sign(np.real(ncon([np.conj(D1temp),\
        np.conj(U),Z,U,D1temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        acc2 = acc2 -(-1.0+np.sign(np.real(ncon([np.conj(D2temp),\
        np.conj(U),Z,U,D2temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        l+=1
    Accuracy = (acc1+acc2)/(4*N)
    return Cost,Cost2,Accuracy


def get_ImproveU1(N,U,D1,D2):
    """ This attempts to improve the stacking using polar decomp of modified cost function"""
    f=100
    A=10
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    m = 1
    while m < 50:
        dz1 = np.zeros(([2,2]),dtype=complex)
        dz2 = np.zeros(([2,2]),dtype=complex)
        l = 1
        while l < N+1:
            D1temp = D1[l-1,:]
            D2temp = D2[l-1,:]
            c1 = np.real(ncon([np.conj(D1temp),np.conj(U),Z,U,D1temp]\
            ,([1],[2,1],[3,2],[3,4],[4])))
            c2 = np.real(ncon([np.conj(D2temp),np.conj(U),Z,U,D2temp]\
            ,([1],[2,1],[3,2],[3,4],[4])))
            """c1 = np.sign(c1)*(np.absolute(c1))**(-2/3)"""
            """c2 = np.sign(c2)*(np.absolute(c2))**(-2/3)"""
            c1 = A/(np.tanh(A)*(np.cosh(A*np.sign(c1)*(np.absolute(c1))))**2)
            c2 = A/(np.tanh(A)*(np.cosh(A*np.sign(c2)*(np.absolute(c2))))**2)
            dz1 = dz1+c1*ncon([np.conj(D1temp),Z,U,D1temp],([-2],[-1,1],[1,2],[2]))
            dz2 =dz2+c2*ncon([np.conj(D2temp),Z,U,D2temp],([-2],[-1,1],[1,2],[2]))
            l +=1
        dZ= (dz1-dz2)
        dZ = dZ/np.sqrt(ncon([dZ,np.conj(dZ)],([1,2],[1,2])))
        U = get_Polar(dZ+f*U)
        """print(get_Classify1(N,U,D1,D2))"""
        m +=1
    return U

def get_Classify2(N,Z2,FF1,FF2):
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    ZI = ncon([Z,eye(2)],([-1,-3],[-2,-4])).reshape([2*2,2*2])
    """ prob of right for 1 minus prob of wrong """
    z1 = np.real(ncon([np.conj(FF1),np.conj(Z2),ZI,Z2,FF1]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/(2*N)
    """ prob of right for 1 minus prob of wrong """
    z2 = np.real(ncon([np.conj(FF2),np.conj(Z2),ZI,Z2,FF2]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/(2*N)
    Cost = z1-z2
    """ Now calculate an equivalent of the training accuracy """
    l=1
    acc1 = 0.0
    acc2 = 0.0
    while l < N+1:
        FF1temp = FF1[l-1,:]
        FF2temp = FF2[l-1,:]
        acc1 = acc1 + (1.0+np.sign(np.real(ncon([np.conj(FF1temp),np.conj(Z2),\
        ZI,Z2,FF1temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        acc2 = acc2 -(-1.0+np.sign(np.real(ncon([np.conj(FF2temp),np.conj(Z2),\
        ZI,Z2,FF2temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        l+=1
    Accuracy = (acc1+acc2)/(4*N)
    return Cost, Accuracy

def get_Z2(N,U,D1,D2):
    """ This attempts to improve the stacking using polar decomp"""
    """ for a single copy nothing changes """
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    II = eye(4)
    ZI = ncon([Z,eye(2)],([-1,-3],[-2,-4])).reshape([2*2,2*2])
    Z2 = eye(4)
    """ initialise Z2 to U*I"""
    Z2 = eye(4)
    """Z2 = ncon([U,eye(2)],([-1,-3],[-2,-4])).reshape([2*2,2*2])"""
    """ Apply improved feature selector then copy data """
    F1 = ncon([U,D1],([-2,1],[-1,1]))
    F2 = ncon([U,D2],([-2,1],[-1,1]))
    FF1 = np.zeros([N,4],dtype=complex)
    FF2 = np.zeros([N,4],dtype=complex)
    l = 1
    while l < N+1:
        tempF1 =  F1[l-1,:]
        tempF2 =  F2[l-1,:]
        FF1[l-1,:] = ncon([tempF1,tempF1],([-1],[-2])).reshape([2*2])
        FF2[l-1,:] = ncon([tempF2,tempF2],([-1],[-2])).reshape([2*2])
        l +=1
    """ construct circuits for derivative of modified cost function"""
    A=10
    f = 50
    m = 1
    while m < 50:
        dz1 = np.zeros(([4,4]),dtype=complex)
        dz2 = np.zeros(([4,4]),dtype=complex)
        C=0
        l = 1
        while l < N+1:
            F1temp = FF1[l-1,:]
            F2temp = FF2[l-1,:]
            c1 = np.real(ncon([np.conj(F1temp),np.conj(Z2),ZI,Z2,F1temp]\
            ,([1],[2,1],[2,3],[3,4],[4])))
            c2 = np.real(ncon([np.conj(F2temp),np.conj(Z2),ZI,Z2,F2temp]\
            ,([1],[2,1],[2,3],[3,4],[4])))
            """c1 = np.sign(c1)*(np.absolute(c1))**(-2/3)
            c2 = np.sign(c2)*(np.absolute(c2))**(-2/3)"""
            c1 = A/(np.tanh(A)*(np.cosh(A*np.sign(c1)*(np.absolute(c1))))**2)
            c2 = A/(np.tanh(A)*(np.cosh(A*np.sign(c2)*(np.absolute(c2))))**2)
            dz1 = dz1 + c1*ncon([np.conj(F1temp),ZI,Z2,F1temp],([-2],[-1,1],[1,2],[2]))
            dz2 = dz2 + c2*ncon([np.conj(F2temp),ZI,Z2,F2temp],([-2],[-1,1],[1,2],[2]))
            C = C + c1-c2
            l +=1
        dZ= (dz1-dz2)
        dZ = dZ/np.sqrt(ncon([dZ,np.conj(dZ)],([1,2],[1,2])))
        Z2 = get_Polar(dZ+f*Z2)
        """print(get_Classify2(N,Z2,FF1,FF2))"""
        m +=1
    print(get_Classify2(N,Z2,FF1,FF2),"Two Copies")
    return Z2

def get_Classify3(N,Z3,FFF1,FFF2):
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    ZII = ncon([Z,eye(2),eye(2)],([-1,-4],[-2,-5],[-3,-6])).reshape([2*2*2,2*2*2])
    """ prob of right for 1 minus prob of wrong """
    c1 = np.real(ncon([np.conj(FFF1),np.conj(Z3),ZII,Z3,FFF1]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/(2*N)
    """ prob of right for 1 minus prob of wrong """
    c2 = np.real(ncon([np.conj(FFF2),np.conj(Z3),ZII,Z3,FFF2]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/(2*N)
    Cost = c1-c2
    """ Now calculate an equivalent of the training accuracy """
    l=1
    acc1 = 0.0
    acc2 = 0.0
    while l < N+1:
        FFF1temp = FFF1[l-1,:]
        FFF2temp = FFF2[l-1,:]
        acc1 = acc1 + (1.0+np.sign(np.real(ncon([np.conj(FFF1temp),np.conj(Z3),\
        ZII,Z3,FFF1temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        acc2 = acc2 -(-1.0+np.sign(np.real(ncon([np.conj(FFF2temp),np.conj(Z3),\
        ZII,Z3,FFF2temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        l+=1
    Accuracy = (acc1+acc2)/(4*N)
    return Cost,Accuracy

def get_Z3(N,U,Z2,D1,D2):
    """ This attempts to improve the stacking using polar decomp"""
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    III = eye(8)
    ZII = ncon([Z,eye(2),eye(2)],([-1,-4],[-2,-5],[-3,-6])).reshape([2*2*2,2*2*2])
    """ initialise Z3 to Z2*I"""
    """Z3 = eye(8)"""
    Z3 = ncon([Z2,eye(2)],([-1,-3],[-2,-4])).reshape([4*2,4*2])
    """ Apply improved feature selector then copy data """
    F1 = ncon([U,D1],([-2,1],[-1,1]))
    F2 = ncon([U,D2],([-2,1],[-1,1]))
    FFF1 = np.zeros([N,8],dtype=complex)
    FFF2 = np.zeros([N,8],dtype=complex)
    A =10
    f=50
    l = 1
    while l < N+1:
        tempF1 =  F1[l-1,:]
        tempF2 =  F2[l-1,:]
        FFF1[l-1,:] = ncon([tempF1,tempF1,tempF1],([-1],[-2],[-3])).reshape([2*2*2])
        FFF2[l-1,:] = ncon([tempF2,tempF2,tempF2],([-1],[-2],[-3])).reshape([2*2*2])
        l +=1
    """ construct circuits for derivative of modified cost function"""
    m = 1
    while m < 50:
        dz1 = np.zeros(([8,8]),dtype=complex)
        dz2 = np.zeros(([8,8]),dtype=complex)
        C=0
        l = 1
        while l < N+1:
            F1temp = FFF1[l-1,:]
            F2temp = FFF2[l-1,:]
            c1 = np.real(ncon([np.conj(F1temp),np.conj(Z3),ZII,Z3,F1temp]\
            ,([1],[2,1],[2,3],[3,4],[4])))
            c2 = np.real(ncon([np.conj(F2temp),np.conj(Z3),ZII,Z3,F2temp]\
            ,([1],[2,1],[2,3],[3,4],[4])))
            """c1 = np.sign(c1)*(np.absolute(c1))**(-2/3)
            c2 = np.sign(c2)*(np.absolute(c2))**(-2/3)"""
            c1 = A/(np.tanh(A)*(np.cosh(A*np.sign(c1)*(np.absolute(c1))))**2)
            c2 = A/(np.tanh(A)*(np.cosh(A*np.sign(c2)*(np.absolute(c2))))**2)
            dz1 = dz1 + c1*ncon([np.conj(F1temp),ZII,Z3,F1temp],([-2],[-1,1],[1,2],[2]))
            dz2 = dz2 + c2*ncon([np.conj(F2temp),ZII,Z3,F2temp],([-2],[-1,1],[1,2],[2]))
            C = C + c1-c2
            l +=1
        dZ= (dz1-dz2)
        dZ = dZ/np.sqrt(ncon([dZ,np.conj(dZ)],([1,2],[1,2])))
        Z3 = get_Polar(dZ+f*Z3)
        """print(get_Classify3(N,Z3,FFF1,FFF2))"""
        m +=1
    print(get_Classify3(N,Z3,FFF1,FFF2),"Three Copies")
    return Z3

def get_Classify4(N,Z4,FFFF1,FFFF2):
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    ZIII = ncon([Z,eye(2),eye(2),eye(2)],([-1,-5],[-2,-6],[-3,-7],[-4,-8])).reshape([2*2*2*2,2*2*2*2])
    """ prob of right for 1 minus prob of wrong """
    c1 = np.real(ncon([np.conj(FFFF1),np.conj(Z4),ZIII,Z4,FFFF1]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/(2*N)
    """ prob of right for 1 minus prob of wrong """
    c2 = np.real(ncon([np.conj(FFFF2),np.conj(Z4),ZIII,Z4,FFFF2]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/(2*N)
    Cost = c1-c2
    """ Now calculate an equivalent of the training accuracy """
    l=1
    acc1 = 0.0
    acc2 = 0.0
    while l < N+1:
        FFFF1temp = FFFF1[l-1,:]
        FFFF2temp = FFFF2[l-1,:]
        acc1 = acc1 + (1.0+np.sign(np.real(ncon([np.conj(FFFF1temp),np.conj(Z4),\
        ZIII,Z4,FFFF1temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        acc2 = acc2 -(-1.0+np.sign(np.real(ncon([np.conj(FFFF2temp),np.conj(Z4),\
        ZIII,Z4,FFFF2temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        l+=1
    Accuracy = (acc1+acc2)/(4*N)
    return Cost,Accuracy

def get_Z4(N,U,Z3,D1,D2):
    """ This attempts to improve the stacking using polar decomp"""
    """ for a single copy nothing changes """
    A=10
    f =10
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    IIII = eye(16)
    ZIII = ncon([Z,eye(2),eye(2),eye(2)],([-1,-5],[-2,-6],[-3,-7],[-4,-8])).reshape([2*2*2*2,2*2*2*2])
    """ initialise Z4 to Z3*I"""
    """Z4 = eye(16)"""
    Z4 = ncon([Z3,eye(2)],([-1,-3],[-2,-4])).reshape([8*2,8*2])
    """ Apply improved feature selector then copy data """
    F1 = ncon([U,D1],([-2,1],[-1,1]))
    F2 = ncon([U,D2],([-2,1],[-1,1]))
    FFFF1 = np.zeros([N,16],dtype=complex)
    FFFF2 = np.zeros([N,16],dtype=complex)
    l = 1
    while l < N+1:
        tempF1 =  F1[l-1,:]
        tempF2 =  F2[l-1,:]
        FFFF1[l-1,:] = ncon([tempF1,tempF1,tempF1,tempF1],([-1],[-2],[-3],[-4])).reshape([2*2*2*2])
        FFFF2[l-1,:] = ncon([tempF2,tempF2,tempF2,tempF2],([-1],[-2],[-3],[-4])).reshape([2*2*2*2])
        l +=1
    """ construct circuits for derivative of modified cost function"""
    m = 1
    while m < 100:
        dz1 = np.zeros(([16,16]),dtype=complex)
        dz2 = np.zeros(([16,16]),dtype=complex)
        l = 1
        while l < N+1:
            F1temp = FFFF1[l-1,:]
            F2temp = FFFF2[l-1,:]
            c1 = np.real(ncon([np.conj(F1temp),np.conj(Z4),ZIII,Z4,F1temp]\
            ,([1],[2,1],[2,3],[3,4],[4])))
            c2 = np.real(ncon([np.conj(F2temp),np.conj(Z4),ZIII,Z4,F2temp]\
            ,([1],[2,1],[2,3],[3,4],[4])))
            """c1 = np.sign(c1)*(np.absolute(c1))**(-2/3)
            c2 = np.sign(c2)*(np.absolute(c2))**(-2/3)"""
            c1 = A/(np.tanh(A)*(np.cosh(A*np.sign(c1)*(np.absolute(c1))))**2)
            c2 = A/(np.tanh(A)*(np.cosh(A*np.sign(c2)*(np.absolute(c2))))**2)
            dz1 = dz1 + c1*ncon([np.conj(F1temp),ZIII,Z4,F1temp],([-2],[-1,1],[1,2],[2]))
            dz2 = dz2 + c2*ncon([np.conj(F2temp),ZIII,Z4,F2temp],([-2],[-1,1],[1,2],[2]))
            l +=1
        dZ= (dz1-dz2)
        dZ = dZ/np.sqrt(ncon([dZ,np.conj(dZ)],([1,2],[1,2])))
        Z4 = get_Polar(dZ+f*Z4)
        """print(get_Classify4(N,Z4,FFFF1,FFFF2))"""
        m +=1
    print(get_Classify4(N,Z4,FFFF1,FFFF2),"Four copies")
    return Z4

def get_Features(N,D1,D2):
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

def get_GreatCirclePoints(U):
    """ get two points on a great circle from U"""
    Theta = 2*np.arctan(np.abs(U[0,1]/U[0,0]))
    Phi = np.angle(U[0,1]/U[0,0])
    p1 = [-np.sin(Phi),np.cos(Phi),0]
    p2 = [np.cos(Theta)*np.cos(Phi),np.cos(Theta)*np.sin(Phi),-np.sin(Theta)]
    return p1,p2

if __name__ == "__main__":
    #np.random.seed(1)
    np.random.seed(1)
    N = 1000
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
    """ gives best test accuracy at theta 1.64 """

    """Initial guess """
    D1 = get_Data(N,Theta1,phi1,chi1,dx1,dy1)
    D2 = get_Data(N,Theta2,phi2,chi2,dx2,dy2)
    U = get_Features(N,D1,D2)
    C = get_Classify1(N,U,D1,D2)
    print(C,"Initial Guess")
    """Improved Guess """
    Uimproved = get_ImproveU1(N,U,D1,D2)
    C = get_Classify1(N,Uimproved,D1,D2)
    print(C,"Improved single stack guess")
    """ exact optium of optimum of original costfunction """
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    M = ncon([np.conj(D1),D1],([1,-2],[1,-1]))\
    - ncon([np.conj(D2),D2],([1,-2],[1,-1]))
    L,W = np.linalg.eig(M)
    X=np.zeros([2,2])
    X[0,1]=1.0
    X[1,0]=1.0
    W = ncon([X,np.conj(W).T],([-1,1],[1,-2]))
    C = get_Classify1(N,W,D1,D2)
    print(C,"Optimal for old cost function")
    """ 2 copies """
    Z2 = get_Z2(N,Uimproved,D1,D2)
    Z3 = get_Z3(N,U,Z2,D1,D2)
    Z4 = get_Z4(N,U,Z3,D1,D2)
