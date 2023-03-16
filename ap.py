import numpy as np
import sklearn.cluster
import sklearn.decomposition
import scipy.optimize
import itertools

def findAllMaps (n):
    maps = []
    l = set(range(n))
    for i in range(n+1):
        for thoseWithCombos in itertools.combinations(l, i):
            remaining = l - set(thoseWithCombos)
            remainingCombos = itertools.combinations(remaining, 2)
            mRemaining = { r:(r,) for r in remaining }
            for assignment in itertools.permutations(remainingCombos, i):
                mCombos = { thoseWithCombos[i]:assignment[i] for i in range(i) }
                maps.append({ **mRemaining, **mCombos })
    return maps

def score (E, y, combos, clusterCombos):
    comparisons = []
    comparisonsHat = []
    for i in range(len(y)):
        for j in range(len(y)):
            if i != j:
                comparisons.append(set(clusterCombos[y[i]]).issubset(clusterCombos[y[j]]))
                comparisonsHat.append(set(combos[E[i]]).issubset(combos[E[j]]))
    comparisons = np.array(comparisons)
    comparisonsHat = np.array(comparisonsHat)
    return np.mean(comparisons == comparisonsHat)

def baselineMethod (E, clusterCenters, y, S, combos, comboInvMap, clusterCombos):
    comboToIdx = {}
    for i, combo in enumerate(combos):
        comboToIdx[combo] = i

    unique, inverse = np.unique(E, return_inverse=True)
    allMaps = findAllMaps(len(unique))
    bestScore = - np.inf
    print("finding best")
    for validMap in allMaps:
        newE = np.array([ comboToIdx[tuple(sorted(clusterCenters[j] for j in validMap[e]))] for e in E ])
        theScore = S[np.arange(len(E)), newE].sum()
        if theScore > bestScore:
            bestScore = theScore
            bestE = newE
            print(validMap)
            print(bestE)
    return bestE

# Example:
# olllllllloooooooollllllllllllllllllllllllllll
# lolllllllolllllllooooooolllllllllllllllllllll
# llolllllllollllllolllllloooooolllllllllllllll
# lllolllllllollllllolllllolllllooooollllllllll
# llllolllllllollllllolllllollllolllloooollllll
# lllllolllllllollllllolllllollllolllolllooolll
# llllllolllllllollllllolllllollllolllollollool
# lllllllolllllllollllllolllllollllolllollololo
# llllllllolllllllollllllolllllollllolllolloloo
def makeMapsForMaxesForSubsets (N, D=2):
    prods = []
    for i in range(D):
        prods += list(itertools.product(range(N), repeat=i+1))
    actualIdxsMap = {}
    usedIdxsMap = {}
    lenSoFar = 0
    for startIdx in range(0, len(prods), N):
        prodsArray = np.array(prods[startIdx:startIdx+N])
        idxs = np.nonzero(np.all(prodsArray[:,0:-1] < prodsArray[:,1:], axis=1))[0]
        prodsArray = prodsArray[idxs]
        # Make sure the indices are monotonically increasing (so that no index is repeated)
        numValidIdxsThisBatch = len(idxs)
        actualIdxsMap[startIdx] = lenSoFar + np.arange(numValidIdxsThisBatch)
        lenSoFar += numValidIdxsThisBatch
        usedIdxsMap[startIdx] = {}
        for k in range(N):
            # Check which indices of prodsArray contain index k
            usedIdxsMap[startIdx][k] = np.nonzero(np.sum(np.isin(prodsArray, k), axis=1) > 0)[0]
        startIdx += N
    return actualIdxsMap, usedIdxsMap, len(prods)

def maxesForSubsets (x, actualIdxsMap, usedIdxsMap, numProds, comboInvMap, N):
    maxes = np.zeros((N,)) - np.inf
    for startIdx in range(0, numProds, N):
        actualIdxs = actualIdxsMap[startIdx]
        actualX = x[actualIdxs]
        if len(actualIdxs) > 0:
            idx1, val1, val2 = argmax2(actualX)  # Note that val2 can be -inf if len(actualX)==1
            for k in range(N):
                usedIdxs = usedIdxsMap[startIdx][k]
                if len(usedIdxs) == 0:
                    maxes[k] = max(maxes[k], val1)
                elif len(usedIdxs) == 1:
                    maxes[k] = max(maxes[k], val2 if idx1 in usedIdxs else val1)
                else:  # len(actualX)-1
                    pass

    # Finally, handle the "core region" for each k (i.e., combos in which k takes part)
    maxesInCoreRegions = np.zeros((N,)) - np.inf
    for k in range(N):
        maxesInCoreRegions[k] = np.max(x[comboInvMap[k]])

    return maxes, maxesInCoreRegions

# Return the top-2 largest elements
def argmax2 (x):
    idx1 = np.argmax(x)
    val1 = x[idx1]
    x[idx1] = -np.inf
    idx2 = np.argmax(x)
    val2 = x[idx2]
    x[idx1] = val1
    return idx1, val1, val2  # Note that we want to allow val2==-inf if len(x)==1

ITERS = 60
LAMBDA = 0.85

def doExceptE (a, k, op, axis):  # op *except* over indices k
    n = a.shape[axis] if len(a.shape) > 1 else a.shape[0]
    idxs = np.nonzero(np.isin(np.arange(n), k, invert=True))[0]
    if len(a.shape) == 1:
        return op(a[idxs])
    elif axis == 0:
        return op(a[idxs,:], axis=0)
    else:
        return op(a[:,idxs], axis=1)

def maxE (a, k):  # max *except* over indices k
    return doExceptE(a, k, np.max, 1)

def sumE (a, k):  # sum *except* over indices k
    return doExceptE(a, k, np.sum, 0)

def rho (S, T, U, combos, comboInvMap, N):
    maxK = np.zeros((N, N))
    maxKE = np.zeros((N, N))
    diag = np.zeros((N,))
    actualIdxsMap, usedIdxsMap, numProds = makeMapsForMaxesForSubsets(N)
    for i in range(N):
        maxesRest, maxesOfComboInv = maxesForSubsets(S[i,:] + T[i,:], actualIdxsMap, usedIdxsMap, numProds, comboInvMap, N)
        maxK[i,:] = np.maximum(maxesOfComboInv - U[i,:,0], maxesRest - U[i,:,1])
        maxKE[i,:] = maxesRest - U[i,:,1]
        diag[i] = S[i,i] + T[i,i] - (U[i,i,0] if i in comboInvMap[i] else U[i,i,1])
    return maxK, maxKE, diag

def alpha (maxK, maxKE, diag, combos, comboInvMap, N):
    T = np.zeros((N, len(combos)))
    U = np.zeros((N, N, 2))
    allIdxsSet = set(np.arange(len(combos)))

    cacheSumEMax = {}
    cacheMaxE = {}
    cacheMax = {}
    cacheSumEMaxE = {}
    cacheIdxs = {}
    for k in range(N):
        cacheMax[k] = maxK[:,k]
        cacheMaxE[k] = maxKE[:,k]
        cacheSumEMax[k] = sumE(cacheMax[k], k)
        cacheSumEMaxE[k] = sumE(cacheMaxE[k], k)
        cacheIdxs[k] = np.array(list(allIdxsSet - set(comboInvMap[k])))

    for i in range(N):
        A1 = np.zeros((N,))
        A2 = np.zeros((N,))
        T2 = 0
        for k in range(N):
            if i == k:
                A1[k] = cacheSumEMax[k]
                A2[k] = cacheSumEMaxE[k]  # c_i does not reference i
            else:
                A1[k] = diag[k] + cacheSumEMax[k] - cacheMax[k][i]
                A2[k] = max(cacheMaxE[k][k] + cacheSumEMaxE[k] - cacheMaxE[k][i], diag[k] + cacheSumEMax[k] - cacheMax[k][i])
            T2 += A2[k]
        #T[i,:] = T2
        for k in range(N):  # Since it operates over the union_k comboInvMap[k], it should be O(len(combos)) in total.
            T[i,comboInvMap[k]] = T[i,comboInvMap[k]] - A2[k] + A1[k]
            U[i,k,:] = (A1[k], A2[k])
    return T, U

def CAP (S, lam, combos, comboInvMap, N):
    maxK = np.zeros((N, N))
    maxKE = np.zeros((N, N))
    diag = np.zeros(N)
    T = np.zeros((N, len(combos)))
    U = np.zeros((N, N, 2))
    little = np.min(S[np.isfinite(S)])
    big = np.max(S[np.isfinite(S)])
    S += 1e-12*np.random.randn(*S.shape)*(big - little)
    for it in range(ITERS):
        if it % 10 == 0:
            print("{}: {}".format(it, len(np.unique(np.argmax(T + S, axis=1)))))
        maxKOld = maxK
        maxKEOld = maxKE
        diagOld = diag
        maxK, maxKE, diag = rho(S, T, U, combos, comboInvMap, N)
        maxK = (1 - lam)*maxK + lam*maxKOld
        maxKE = (1 - lam)*maxKE + lam*maxKEOld
        diag = (1 - lam)*diag + lam*diagOld

        TOld = T
        UOld = U
        T, U = alpha(maxK, maxKE, diag, combos, comboInvMap, N)
        T -= np.min(T[np.isfinite(T)])
        T = (1 - lam)*T + lam*TOld
        U = (1 - lam)*U + lam*UOld
    E = np.argmax(T + S, axis=1)
    print("{}: {}".format(it, len(np.unique(np.argmax(T + S, axis=1)))))
    return E

def makeCombos (N):
    combos = []
    comboInvMap = {}

    for i in range(N):
        comboInvMap[i] = []
        combos.append((i,))
        comboInvMap[i].append(len(combos) - 1)

    for i in range(N-1):
        for j in range(i + 1, N):
            combos.append((i,j))
            comboInvMap[i].append(len(combos) - 1)
            comboInvMap[j].append(len(combos) - 1)
    return combos, comboInvMap
