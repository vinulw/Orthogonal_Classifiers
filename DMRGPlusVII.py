from scipy import integratefrom scipy.io import loadmatfrom ncon import nconimport matplotlib.pyplot as pltfrom scipy.linalg import expmimport scipy as spfrom pylab import *import numpy as npfrom scipy.linalg import eigimport pylab as plfrom scipy import integrate""" Conventions:This code tries to optimise V^dagger M V with M hermitian and V unitary  """# def get_MV(Mdata, D, m):#     """ Returns a structured M corresponding to m copies and the best V """##     V = np.zeros((D ** m, D ** m), dtype=np.float64)#     Vr = np.zeros((D, D ** m), dtype=np.float64)#     Udata = np.zeros((D, D, D), dtype=np.float64)#     Ldata = np.zeros((D, D), dtype=np.float64)#     ULPm = np.zeros((D ** m, D, D), dtype=np.float64)#     ELPm = np.zeros((D ** m, D, D), dtype=np.float64)#     ELLm = np.zeros((D ** m, D, D), dtype=np.float64)#     M = np.zeros((D ** m * D ** m, D ** m * D ** m), dtype=np.float64)##     """ a) First we strip out the SVD data from Mdata """##     for l in range(D):#         MI = Mdata[:, :, l]##         UI, LI, VI = np.linalg.svd(MI)##         Udata[:, :, l] = UI#         Ldata[:, l] = LI##     """ b) Next, construct the higher dimensional vectors ulp and elp """##     """ Udata[i,p,l] - u[i,l,p] in notation from notes """##     """ ELP[index,p,l] - |el ep^m>  in notation from notes """##     """ NB: the ordering is swapped in ELP to |ep^m el > """##     for l in range(D):##         el = eye(D)[:, l]##         for p in range(D):##             ULP1 = Udata[:, p, l]##             ULP = ULP1##             ep = eye(D)[:, p]##             ELL, ELP = el, el##             for k in range(1, m):#                 ELL = np.kron(el, ELL)##                 ULP = np.kron(ULP1, ULP)##                 ELP = np.kron(ep, ELP)##             ULPm[:, p, l] = ULP##             ELPm[:, p, l] = ELP##             ELLm[:, p, l] = ELL##     """print((ncon([np.conj(Udata[:,0,0]),Udata[:,0,1]],([1],[1])))**m)#     print(ncon([np.conj(ULPm[:,0,0]),ULPm[:,0,1]],([1],[1])))"""##     """ b) Construct enlarged M """##     for l in range(D):#         for p in range(D):#             for q in range(D):#                 EULPQm = np.kron(ELPm[:, q, l], ULPm[:, p, l])##                 M = M + Ldata[p, l] * np.outer(EULPQm, np.transpose(np.conj(EULPQm)))##     """ c) Construct V: """##     """ This is a non-unitary matrix that saturates C. I have tried  """#     """ constructing this and using various polar decompositions. It """#     """ turns out that DMRG is better than anything that I have been """#     """ able to do with this however """#     """ There are two constructions here with |el ep><ulp ulp|  or """#     """ with |el el> <ulp ulp| They both saturate V, but polar """#     """decompositions are poor in both cases. This can be ignored really """##     for l in range(D):#         for p in range(D):#             """V = V + np.outer(ELPm[:,p-1,l-1],ULPm[:,p-1,l-1])"""##             V = V + np.outer(ELLm[:, p, l], ULPm[:, p, l])##     """ d) Construct rectangular V """##     for l in range(D):#         for p in range(D):#             Vr = Vr + np.outer(eye(D)[:, l], ULPm[:, p, l])##     return M, V, Vrdef init_V(D):    """ Returns a random unitary """    return np.linalg.qr(np.random.rand(D, D))[0]def get_Mdata(D):    """ returns a tensor containing D random DxD hermitian matrice """    """ unfortunately this is not good enough as we can get negative"""    """ Schmidt coefficients (actually compensated by a change in V)"""    """ The version that is actually run creates a random Hermitian """    """ with real singular values """    """l=1    Mdata = np.zeros((D,D,D),dtype=np.float64)    while l < D+1:        A = np.random.rand(D,D)        Mdata[:,:,l-1] = A + np.transpose(np.conj(A))        Mdata[:,:,l-1] = Mdata[:,:,l-1]/np.trace(Mdata[:,:,l-1])        l +=1"""    """ Generate Mdata instead from a random unitary and a random positive vector """    Mdata = np.zeros((D, D, D), dtype=np.float64)    for l in range(D):        U = init_V(D)        randomvec = init_V(D)[:, 0]        L = np.diag(np.conj(randomvec) * randomvec)        Mdata[:, :, l] = ncon([U, L, U.conj().T], ([-1, 1], [1, 2], [2, -2]))        Mdata[:, :, l] = Mdata[:, :, l] / np.trace(Mdata[:, :, l])    return Mdatadef get_M(Mdata, D, L, P, m):    """ Returns a structured M corresponding to m copies and the best V """    Udata = np.zeros((D, P, L), dtype=np.float64)    Ldata = np.zeros((D, L), dtype=np.float64)    [ULPm, ELPm, ELLm] = [np.zeros((D ** m, D, D), dtype=np.float64) for _ in range(3)]    M = np.zeros((D ** m * D ** m, D ** m * D ** m), dtype=np.float64)    """ a) First we strip out the SVD data from Mdata """    for l in range(L):        MI = Mdata[:, :, l]        UI, LI, VI = np.linalg.svd(MI)        Udata[:, :, l] = UI        Ldata[:, l] = LI    """ b) Next, construct the higher dimensional vectors ulp and elp """    """ Udata[i,p,l] - u[i,l,p] in notation from notes """    """ ELP[index,p,l] - |el ep^m>  in notation from notes """    """ NB: the ordering is swapped in ELP to |ep^m el > """    for l in range(L):        el = eye(D)[:, l]        for p in range(P):            ULP1 = Udata[:, p, l]            ULP = ULP1            ep = eye(D)[:, p]            ELL, ELP = el, el            for k in range(1, m):                ELL = np.kron(el, ELL)                ULP = np.kron(ULP1, ULP)                ELP = np.kron(ep, ELP)            ULPm[:, p, l] = ULP            ELPm[:, p, l] = ELP            ELLm[:, p, l] = ELL    """print((ncon([np.conj(Udata[:,0,0]),Udata[:,0,1]],([1],[1])))**m)    print(ncon([np.conj(ULPm[:,0,0]),ULPm[:,0,1]],([1],[1])))"""    """ b) Construct enlarged M """    for l in range(L):        for p in range(P):            for q in range(P):                EULPQm = np.kron(ELPm[:, q, l], ULPm[:, p, l])                M = M + Ldata[p, l] * np.outer(EULPQm, EULPQm.conj().T)    return M# def get_M(Mdata, D, m):#     """ Returns a structured M corresponding to m copies and the best V """##     Udata = np.zeros((D, D, D), dtype=np.float64)#     [ULPm, ELPm] = [np.zeros((D ** m, D), dtype=np.float64) for _ in range(2)]#     M = np.zeros((D ** m * D ** m, D ** m * D ** m), dtype=np.float64)##     """ a) First we strip out the SVD data from Mdata """##     for l in range(D):#         MI = Mdata[:, :, l]##         UI, LI, VI = np.linalg.svd(MI)##         Udata[:, l] = UI[0]##     """ b) Next, construct the higher dimensional vectors ulp and elp """##     """ Udata[i,p,l] - u[i,l,p] in notation from notes """##     """ ELP[index,p,l] - |el ep^m>  in notation from notes """##     """ NB: the ordering is swapped in ELP to |ep^m el > """##     for l in range(D):##         ULP = Udata[:, l]#         ELP = eye(D)[:, l]##         for k in range(1, m):#             ULP = np.kron(Udata[:, l], ULP)#             ELP = np.kron(eye(D)[:, 0], ELP)##         ULPm[:, l] = ULP#         ELPm[:, l] = ELP##     """print((ncon([np.conj(Udata[:,0,0]),Udata[:,0,1]],([1],[1])))**m)#     print(ncon([np.conj(ULPm[:,0,0]),ULPm[:,0,1]],([1],[1])))"""##     """ b) Construct enlarged M """##     for l in range(D):#         EULPQm = np.kron(ELPm[:, l], ULPm[:, l])#         M = M + np.outer(EULPQm, EULPQm.conj().T)##     return Mdef get_VDMRGPlus(M, d):    """ Returns V evaluated from the R fixed point of M   """    """ This never worked particularly well. The explicit """    """ construction of a non-unitary V above shows that there """    """ is a huge degeneracy of the fixed point. """    M = M.reshape([d * d, d * d])    e, R = eig(M, k=1, which='LM')    R = R.reshape([d, d])    X, L, Y = np.linalg.svd(R)    VDMRGPlus = ncon([X, Y], ([-1, 1], [1, -2]))    return VDMRGPlusdef get_VDMRG(Vseed, M, d):    """ Returns V using DMRG: this works best   """    V = Vseed    Niterations = 100    for i in range(1, Niterations):        X, L, Y = np.linalg.svd(ncon([M, V], ([-1, -2, 1, 2], [1, 2])))        """rather than one step we have introduced a finite learning rate """        V = ncon([X, Y], ([-1, 1], [1, -2])) + 2.0 * V        X, L, Y = np.linalg.svd(V)        V = ncon([X, Y], ([-1, 1], [1, -2]))        C = get_C(M, V, d)        print(f'C iteration {i} = {C}')    return Vdef get_C(M, V, d):    """ Calulate Cost Function """    return ncon([np.conj(V), M, V], ([1, 2], [1, 2, 3, 4], [3, 4]))if __name__ == "__main__":    # np.random.seed(1)    D = 2    P = 2    L = 2    mmax = 7    Mdata = get_Mdata(D)    print(D)    for m in range(2, mmax + 1):        # M, _, _ = get_MV(Mdata, D, m)        M = get_M(Mdata, D, P, L, m)        M = M.reshape([D ** m, D ** m, D ** m, D ** m])        """C = get_C(M,V,D**m)        print(C) this is always equal to D so no need to print"""        """Copt = get_optC(M,D**m,50000)        print(Copt)"""        VDMRG = get_VDMRG(eye(D ** m), M, D ** m)        CDMRG = get_C(M, VDMRG, D ** m)        print(CDMRG)def get_optC(M, d, Nsample):    """ get optimum C by random sampling over V"""    optC = 0.0    for i in range(1, Nsample):        """Random sampling"""        V = init_V(d)        C = get_C(M, V, d)        optC = max(optC, C)    return optC