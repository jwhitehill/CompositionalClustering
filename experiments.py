from turtle import forward
import numpy as np
import sys
import os
import sklearn.cluster
import sklearn.decomposition
import scipy.optimize
import itertools
import matplotlib.pyplot as plt
import sklearn.cluster
#from matplotlib.backends.backend_pdf import PdfPages
from ap import CAP, baselineMethod
from multiprocessing import Pool
import skfuzzy as fuzz
from itertools import combinations
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
import torch
import torch.nn as nn
from torch import no_grad, optim
import torch.nn.functional as F
import random
from score_use import score, makeCombos, makeCombos3
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ITERS = 60
LAMBDA = 0.85

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def centroid (S, idxs):
    someS = - S[idxs[:,None],idxs]
    sums = np.zeros(len(idxs))
    for i in range(len(idxs)):
        sums[i] = np.sum(S[i,:]) - S[i,i]
    return np.argmin(sums)

def AC (S, x, y, clusterCombos, thresh, useHeuristic = False):
    print("AC---------")
    N = len(x)
    combos, comboInvMap = makeCombos(N)  # combos of datapoints
    ag = sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=thresh)
    ag.fit(x)
    clusterIndices = {}
    for label in np.unique(ag.labels_):
        idxs = np.nonzero(ag.labels_ == label)[0]
        clusterIndices[label] = idxs[centroid(S, idxs)]
    if useHeuristic and len(np.unique(ag.labels_)) < 8:
        E = baselineMethod(ag.labels_, clusterIndices, y, S, combos, comboInvMap, clusterCombos)
    else:
        E = ag.labels_

    theScore = score(E, y, combos, clusterCombos)
    print("AC best: {}".format(theScore))
    return theScore, E

def APstandard (S, pref, y, clusterCombos, discount, useHeuristic = False):
    N = S.shape[0]
    combos, comboInvMap = makeCombos(N)  # combos of datapoints
    refAP = sklearn.cluster.AffinityPropagation(affinity='precomputed', max_iter=ITERS*5,\
                                                damping=LAMBDA, verbose=True, preference=-pref * discount)
    refAP.fit(S[0:N, 0:N])
    # print(refAP.cluster_centers_indices_)
    if useHeuristic and type(refAP.cluster_centers_indices_) != type([]) and len(refAP.cluster_centers_indices_) < 8:
        E = baselineMethod(refAP.labels_, refAP.cluster_centers_indices_, y, S, combos, comboInvMap, clusterCombos)
    else:
        E = refAP.labels_

    if type(refAP.cluster_centers_indices_) == type([]) and refAP.cluster_centers_indices_ == []:
        theBaselineScore = - np.inf
    else:
        theBaselineScore = score(E, y, combos, clusterCombos)

    # print("baseline score: {}".format(theBaselineScore))
    return theBaselineScore, E

def FuzzyCMeans (x, y, numClusters, clusterCombos, m):
    clusterCombosMap, invClusterCombosMap = makeCombos(numClusters)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(x.T, numClusters, m, error=0.005, maxiter=1000)
    res = u.T
    threshold = 1/numClusters
    pos = res > threshold
    E = []
    for p_cnt, p in enumerate(pos):
        idxs = np.argwhere(p).squeeze(-1)
        numLabel = len(idxs)
        if numLabel == 0:
            idxs = np.argmax(res[p_cnt])
        elif numLabel > 2:
            idxs = (-res)[p_cnt].argsort()[:2]

        if len(idxs) == 2:
            idx0 = invClusterCombosMap[idxs[0]]
            idx1 = invClusterCombosMap[idxs[1]]
            label = list(set(idx0).intersection(set(idx1)))[0]
        else:
            label = idxs[0]
        E.append(label)
    theScore = score(E, y, clusterCombosMap, clusterCombos)

    return theScore, E

def FuzzyCMeans3 (x, y, numClusters, clusterCombos, m):
    clusterCombosMap, invClusterCombosMap = makeCombos3(numClusters)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(x.T, numClusters, m, error=0.005, maxiter=1000)
    res = u.T
    threshold = 1/numClusters
    pos = res > threshold
    E = []
    for p_cnt, p in enumerate(pos):
        idxs = np.argwhere(p).squeeze(-1)
        numLabel = len(idxs)
        if numLabel == 0:
            idxs = [np.argmax(res[p_cnt])]
        elif numLabel > 3:
            idxs = (-res)[p_cnt].argsort()[:3]
        if len(idxs) == 3:
            idx0 = invClusterCombosMap[idxs[0]]
            idx1 = invClusterCombosMap[idxs[1]]
            idx2 = invClusterCombosMap[idxs[2]]
            label = list(set(idx0).intersection(set(idx1)).intersection(set(idx2)))[0]
        if len(idxs) == 2:
            idx0 = invClusterCombosMap[idxs[0]]
            idx1 = invClusterCombosMap[idxs[1]]
            label = list(set(idx0).intersection(set(idx1)))[0]
        else:
            label = idxs[0]
        E.append(label)
    theScore = score(E, y, clusterCombosMap, clusterCombos)

    return theScore, E

import torch.nn as nn
import torch.nn.functional as F
class G_net (nn.Module):
    def __init__ (self):
        super(G_net, self).__init__()
        self.linear1a = nn.Linear(32, 32)
        self.linear1b = nn.Linear(32, 32)

    def forward (self, X1, X2):
        linear1 = self.linear1a(X1) + self.linear1a(X2) + self.linear1b(X1*X2)
        return linear1

def GCA (y, someemb, numClusters, clusterCombos, m, spref2, g_path="omniglot_checkpoint/g_net_no_bn.pt"):

    someembTensor = torch.tensor(someemb)
    someembnorm = F.normalize(someembTensor).data.cpu().numpy()

    # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(someembnorm.T, numClusters, m, error=0.005, maxiter=1000)
    # res = u.T
    # E = []
    # centroids = [[] for _ in range(numClusters)]
    # for r_cnt, r in enumerate(res):
    #     idx = np.argmax(r)
    #     centroids[idx].append(someemb[r_cnt])
    #     E.append(idx)
    # ag = sklearn.cluster.KMeans(n_clusters=numClusters)
    ag = sklearn.cluster.AgglomerativeClustering(n_clusters=numClusters, distance_threshold=None)
    # ag = sklearn.cluster.AffinityPropagation(max_iter=ITERS*2, random_state=None, damping=LAMBDA, preference=-1)
    ag.fit(someembnorm)
    E = ag.labels_
    numClusters = max(E)+1
    centroids = [[] for _ in range(numClusters)]
    for e_cnt, e in enumerate(E):
        centroids[e].append(someemb[e_cnt])

    cs = []
    for c in centroids:
        cs.append(np.mean(np.stack(c), 0))
    cs = torch.tensor(np.stack(cs)).to(device)

    comb = combinations(range(numClusters), 2)
    comb = torch.tensor([[a, b] for a, b in comb]).to(device)

    embPairs = cs[comb].to(device)
    with torch.no_grad():
        g_net = G_net().to(device)
        g_net.load_state_dict(torch.load(g_path))
        g_net.eval()
        embComp = g_net(embPairs[:, 0], embPairs[:, 1])
        embNorm = F.normalize(cs).data.cpu().numpy()
        embCompNorm = F.normalize(embComp).data.cpu().numpy()
    x = embNorm
    X = embCompNorm #np.concatenate([embNorm, embCompNorm], 0)
    S = np.zeros((len(x), len(X)))
    for xCnt in range(len(x)):
        S[xCnt] = np.linalg.norm(x[xCnt, :] - X, axis=1) **2

    for cCnt, c in enumerate(comb):
        S[c[0], cCnt] = np.inf
        S[c[1], cCnt] = np.inf

        
    clusterCombosMap, invClusterCombosMap = makeCombos(numClusters)

    for sample_cnt in range(numClusters):
        for idx in invClusterCombosMap[sample_cnt]:
            if idx >= numClusters:
                S[sample_cnt, idx - numClusters] = np.inf
    
    minIdx = np.argmin(S, 1)
    minVal = S[np.arange(S.shape[0]),minIdx]

    order = np.argsort(minVal)
    
    singletons = set()
    comp = set()
    newE = [i for i in range(numClusters)]
    for o in order:
        val = minVal[o]
        if val > spref2:
            break
        curIdx = minIdx[o]
        curChildren = clusterCombosMap[curIdx + numClusters]
        if comp.intersection(curChildren) or o in singletons:
            break
        newE[o] = curIdx + numClusters
        comp.add(o)
        singletons.update(curChildren)

    # singletons = set()
    # comp = set()
    # newE = [i for i in range(numClusters)]
    # model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    # model = model.fit(X)
    # children = model.children_
    # dis = model.distances_
    # for child, d in zip(children, dis):
    #     if np.amin(child) < numClusters and np.amax(child) >= numClusters and np.amax(child) < len(clusterCombosMap):
    #         newE[np.amin(child)] = np.amax(child)
    #         curChildren = clusterCombosMap[np.amax(child)]
    #         if comp.intersection(curChildren) or np.amin(child) in singletons:
    #             break
    #         comp.add(np.amin(child))
    #         singletons.update(curChildren)
    
    
    
    # spref = - S
    # np.fill_diagonal(spref, -spref2)

    # newE = CAP(spref, LAMBDA, clusterCombosMap, invClusterCombosMap, numClusters)
    # numClasses = len(set(newE))

    finalE = []
    for e in E:
        finalE.append(newE[e])

    theScore = score(finalE, y, clusterCombosMap, clusterCombos)

    return theScore, E

def FuzzyCMeansSub (x, y, numClusters, clusterCombos, m, size):
    selidxs = np.sort(np.random.permutation(x.shape[0])[0:size])
    somex = x[selidxs]
    clusterCombosMap, invClusterCombosMap = makeCombos(numClusters)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(somex.T, numClusters, m, error=0.005, maxiter=1000)
    res = u.T
    threshold = 1/numClusters
    pos = res > threshold
    E = []
    for p_cnt, p in enumerate(pos):
        idxs = np.argwhere(p).squeeze(-1)
        numLabel = len(idxs)
        if numLabel == 0:
            idxs = np.argmax(res[p_cnt])
        elif numLabel > 2:
            idxs = (-res)[p_cnt].argsort()[:2]

        if len(idxs) == 2:
            idx0 = invClusterCombosMap[idxs[0]]
            idx1 = invClusterCombosMap[idxs[1]]
            label = list(set(idx0).intersection(set(idx1)))[0]
        else:
            label = idxs[0]
        E.append(label)
    
    fullE = np.zeros(len(x)).astype(int)
    for i in range(len(x)):
        dist = np.linalg.norm(x[i, :] - somex, axis=1) **2
        EIndx = np.argmin(dist)
        fullE[i] = E[EIndx]

    theScore = score(fullE, y, clusterCombosMap, clusterCombos)

    return theScore, E


def fullCAP (S, x, pref, y, clusterCombos):
    N = len(x)
    combos, comboInvMap = makeCombos(N)  # combos of datapoints
    print("FULL METHOD")
    E = CAP(S, LAMBDA, combos, comboInvMap, N)
    print(E)
    theScore = score(E, y, combos, clusterCombos)
    print("CAP score: {}".format(theScore))
    return theScore, E

def subsetCAPOptimized(someS, somex, x, pref, y, clusterCombos, size, idxs):
    M = size
    someCombos, someComboInvMap = makeCombos(M)
    someS = - someS
    np.fill_diagonal(someS, -pref)
    for i in range(M):
        someS[i,np.array(list(set(someComboInvMap[i]) - set((i,))))] = -np.inf
    someE = CAP(someS, LAMBDA, someCombos, someComboInvMap, M)
    E = np.zeros(len(x))
    for i in range(len(x)):
        dist = np.linalg.norm(x[i, :] - somex, axis=1) **2
        EIndx = np.argmin(dist)
        E[i] = someE[EIndx]
    
    N = len(x)
    combos, _ = makeCombos(N)
    comboToIdx = {}
    for i, combo in enumerate(combos):
        comboToIdx[combo] = i
    remappedE = np.zeros((N,)).astype(np.int32)
    for i in range(len(E)):
        someCombo = someCombos[int(E[i])]
        remappedE[i] = comboToIdx[tuple(sorted([idxs[j] for j in someCombo]))]
    E = remappedE

    print(E)
    theScore = score(E, y, combos, clusterCombos)
    print("CAPsub score: {}".format(theScore))
    return theScore, E

def subsetCAP (S, x, pref, y, clusterCombos, size):
    print("SUBSET METHOD")
    M = size  # random subset
    # Randomly select some examples
    largeCombos, largeComboInvMap = makeCombos(len(x))
    orderDict = {}
    for combIdx, comb in enumerate(largeCombos):
        key = str(comb)
        orderDict[key] = combIdx
    someCombos, someComboInvMap = makeCombos(M)
    idxs = np.sort(np.random.permutation(x.shape[0])[0:M])
    someS = np.zeros((M,int(M + M * (M - 1) / 2 )))
    for rowCnt, rowIdx in enumerate(idxs):
        for comboCnt, combo in enumerate(someCombos):
            if len(combo) == 1:
                key = str((idxs[combo[0]],))
            else:
                key = str((idxs[combo[0]],idxs[combo[1]]))
            colIdx = orderDict[key]
            someS[rowCnt, comboCnt] = S[rowIdx, colIdx]
    # somex = x[idxs]
    # someX = np.array([ np.max(somex[np.array(combo),:], axis=0) for combo in someCombos ])
    # someS = np.linalg.norm((somex[:,np.newaxis,:] - someX), axis=2) ** 2
    someS = - someS
    np.fill_diagonal(someS, -pref)
    for i in range(M):
        someS[i,np.array(list(set(someComboInvMap[i]) - set((i,))))] = -np.inf
    someE = CAP(someS, LAMBDA, someCombos, someComboInvMap, M)
    SwrtExemplars = np.zeros((len(x), M))
    # SwrtExemplars = np.linalg.norm((someX[someE,:] - x[:,np.newaxis,:]), axis=2) **2
    for rowCnt in range(S.shape[0]):
        for comboCnt, comboIdx in enumerate(someE):
            combo = someCombos[comboIdx]
            if len(combo) == 1:
                key = str((idxs[combo[0]],))
            else:
                key = str((idxs[combo[0]],idxs[combo[1]]))
            colIdx = orderDict[key]
            SwrtExemplars[rowCnt, comboCnt] = S[rowCnt, colIdx]

    E = someE[np.argmin(SwrtExemplars, axis=1)]

    # Remap the cluster labels back to the full index space
    N = len(x)
    combos, _ = makeCombos(N)
    comboToIdx = {}
    for i, combo in enumerate(combos):
        comboToIdx[combo] = i
    remappedE = np.zeros((N,)).astype(np.int32)
    for i in range(len(E)):
        someCombo = someCombos[E[i]]
        remappedE[i] = comboToIdx[tuple(sorted([idxs[j] for j in someCombo]))]
    E = remappedE

    print(E)
    theScore = score(E, y, combos, clusterCombos)
    print("CAPsub score: {}".format(theScore))
    return theScore, E

def GMM (y, someemb, numClusters, clusterCombos, threshold, reg_covar):
    clusterCombosMap, invClusterCombosMap = makeCombos(numClusters)
    threshold = 1/numClusters
    gm = BayesianGaussianMixture(n_components=numClusters, reg_covar=reg_covar, max_iter=500).fit(someemb)
    prob = gm.predict_proba(someemb)

    pos = prob > threshold
    E = []
    for p_cnt, p in enumerate(pos):
        idxs = np.argwhere(p).squeeze(-1)
        numLabel = len(idxs)
        if numLabel == 0:
            idxs = [np.argmax(prob[p_cnt])]
        elif numLabel > 2:
            sorted_idx = (-prob)[p_cnt].argsort()
            idxs = sorted_idx[:2]
      
        if len(idxs) == 2:
            idx0 = invClusterCombosMap[idxs[0]]
            idx1 = invClusterCombosMap[idxs[1]]
            label = list(set(idx0).intersection(set(idx1)))[0]
        else:
            label = idxs[0]
        E.append(label)
    theScore = score(E, y, clusterCombosMap, clusterCombos)

    return theScore, E

# def GMM (y, someemb, numClusters, clusterCombos, thredhold, reg_covar):
#     combs = list(combinations(range(numClusters), 2))
#     remap = {}
#     for c_cnt, c in enumerate(combs):
#         remap[str(c)] = c_cnt

#     gm = BayesianGaussianMixture(n_components=numClusters, reg_covar=reg_covar, max_iter=500).fit(someemb)
#     prob = gm.predict_proba(someemb)
#     # import pdb;pdb.set_trace()
#     maxProb = np.max(prob, 1)
#     compositional = maxProb < thredhold
#     clusterCombosMap, invClusterCombosMap = makeCombos(numClusters)

#     E = []
#     for p, c in zip(prob, compositional):
#         if c:
#             sorted_idx = np.argsort(p)
#             minidx = np.min([sorted_idx[-1], sorted_idx[-2]])
#             maxidx = np.max([sorted_idx[-1], sorted_idx[-2]])
#             remapedIdx = remap[f'({minidx}, {maxidx})']+numClusters
#             E.append(remapedIdx)
#         else:
#             E.append(np.argmax(p))
#     theScore = score(E, y, clusterCombosMap, clusterCombos)
#     # import pdb;pdb.set_trace()

#     return theScore, E

def GMM3 (y, someemb, numClusters, clusterCombos, thredhold, reg_covar):
    clusterCombosMap, invClusterCombosMap = makeCombos3(numClusters)
    gm = BayesianGaussianMixture(n_components=numClusters, reg_covar=reg_covar, max_iter=500).fit(someemb)
    prob = gm.predict_proba(someemb)
    thredhold = 1/numClusters
    pos = prob > thredhold
    E = []
    for p_cnt, p in enumerate(pos):
        idxs = np.argwhere(p).squeeze(-1)
        numLabel = len(idxs)
        if numLabel == 0:
            idxs = [np.argmax(prob[p_cnt])]
        elif numLabel > 3:
            sorted_idx = (-prob)[p_cnt].argsort()
            idxs = sorted_idx[:3]
        if len(idxs) == 3:
            idx0 = invClusterCombosMap[idxs[0]]
            idx1 = invClusterCombosMap[idxs[1]]
            idx2 = invClusterCombosMap[idxs[2]]
            label = list(set(idx0).intersection(set(idx1)).intersection(set(idx2)))[0]
        elif len(idxs) == 2:
            idx0 = invClusterCombosMap[idxs[0]]
            idx1 = invClusterCombosMap[idxs[1]]
            label = list(set(idx0).intersection(set(idx1)))[0]
        else:
            label = idxs[0]
        E.append(label)
    theScore = score(E, y, clusterCombosMap, clusterCombos)

    return theScore, E

class DistLoss(nn.Module):
    def __init__(self, init, gPath):
        super(DistLoss, self).__init__()
        self.W = nn.Parameter(torch.tensor(init))
        self.combinations2 = torch.tensor(list(combinations(list(range(len(init))),2)))
        self.g_net = G_net().to(device)
        self.g_net.load_state_dict(torch.load(gPath, map_location=device))

    def forward(self, x):
        # generate compositional centroids: merge2
        comb2_a = self.W[self.combinations2.transpose(-2, -1)[0]]
        comb2_b = self.W[self.combinations2.transpose(-2, -1)[1]]
        merged2 = self.g_net(comb2_a, comb2_b)

        allCentroids = torch.cat([self.W, merged2], 0)

        assignment = torch.argmin(torch.cdist(F.normalize(x), F.normalize(allCentroids)), 1)
        assignCentroids = allCentroids[assignment]
        dists = torch.dist(F.normalize(assignCentroids), F.normalize(x))
        return dists, assignment

def CKM(y, x, numClusters, clusterCombos, gPath):
    
    min_loss = np.inf
    for init_cnt in tqdm(range(100)):
        initCentroids = x[random.sample(range(len(x)), numClusters)]
        computeLoss = DistLoss(initCentroids, gPath).to(device)

        optimizer = optim.Adam([computeLoss.W], lr=0.2)
        xTorch = torch.from_numpy(x).to(device)

        for iterCnt in range(100):
            optimizer.zero_grad()
            loss, assignment = computeLoss(xTorch) 
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            if loss < min_loss:
                min_loss = loss.detach()
                E = assignment.cpu().data.numpy()
    clusterCombosPred, _ = makeCombos(numClusters)
    theScore = score(E, y, clusterCombosPred, clusterCombos)
    return theScore, E

def GCA_CKM(y, someemb, numClusters, clusterCombos, threshold=0.4):
    someembTensor = torch.tensor(someemb)
    someembnorm = F.normalize(someembTensor).data.cpu().numpy()

    ag = sklearn.cluster.AgglomerativeClustering(n_clusters=numClusters, distance_threshold=None)
    ag.fit(someembnorm)
    E = ag.labels_
    numClusters = max(E)+1
    centroids = [[] for _ in range(numClusters)]
    for e_cnt, e in enumerate(E):
        centroids[e].append(someemb[e_cnt])

    cs = []
    for c in centroids:
        cs.append(np.mean(np.stack(c), 0))
    cs_numpy = np.stack(cs)
    cs = torch.tensor(cs_numpy).to(device)

    comb = combinations(range(numClusters), 2)
    comb = torch.tensor([[a, b] for a, b in comb]).to(device)

    embPairs = cs[comb].to(device)
    with torch.no_grad():
        g_net = G_net().to(device)
        g_net.load_state_dict(torch.load("checkpoint_libri3_CAP/0.868_g.pt"))
        # g_net.load_state_dict(torch.load("omniglot_checkpoint/g_net_no_bn.pt"))
        g_net.eval()
        embComp = g_net(embPairs[:, 0], embPairs[:, 1])
        embNorm = F.normalize(cs).data.cpu().numpy()
        embCompNorm = F.normalize(embComp).data.cpu().numpy()
    x = embNorm
    X = embCompNorm #np.concatenate([embNorm, embCompNorm], 0)
    S = np.zeros((len(x), len(X)))
    for xCnt in range(len(x)):
        S[xCnt] = np.linalg.norm(x[xCnt, :] - X, axis=1) **2

    for cCnt, c in enumerate(comb):
        S[c[0], cCnt] = np.inf
        S[c[1], cCnt] = np.inf

        
    clusterCombosMap, invClusterCombosMap = makeCombos(numClusters)

    for sample_cnt in range(numClusters):
        for idx in invClusterCombosMap[sample_cnt]:
            if idx >= numClusters:
                S[sample_cnt, idx - numClusters] = np.inf
    
    minIdx = np.argmin(S, 1)
    minVal = S[np.arange(S.shape[0]),minIdx]

    # order = np.argsort(minVal)

    singleIdx = np.where(minVal > threshold)

    singleCentroids = cs_numpy[singleIdx]
    computeLoss = DistLoss(singleCentroids).to(device)

    optimizer = optim.Adam([computeLoss.W], lr=0.12)
    xTorch = torch.from_numpy(someemb).to(device)
    for iterCnt in range(2000):
        optimizer.zero_grad()
        loss, assignment, dists = computeLoss(xTorch)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        E = assignment.cpu().data.numpy()

    clusterCombosE, _ = makeCombos(len(singleCentroids))
    theScore = score(E, y, clusterCombosE, clusterCombos)
    # print(iterCnt, loss, theScore)
    # a=1
    return theScore, E
    

def experiments (K, NperC, xPath, XPath, originalSPath, Spref, thresh, method, embPath=None, someSPath=None, somexPath=None, idxsPath=None, numClusters = 5, m=1.5, spref2=0.1, g_path=None):
    K = K
    NSample = NperC
    clusterCombos, _ = makeCombos(K)  # combos datapoints
    N = len(clusterCombos) * NSample
    # combos, comboInvMap = makeCombos(N)  # combos of datapoints

    if originalSPath:
        originalS = np.load(originalSPath)
    if xPath:
        x = np.load(xPath)
    if XPath:
        X = np.load(XPath)

    
    y = [i for i in range(len(clusterCombos)) for _ in range(NSample)]

    if originalSPath:
        S = - originalS
        # for i in range(N):
        #     S[i,np.array(list(set(comboInvMap[i]) - set((i,))))] = -np.inf
        np.fill_diagonal(S, -Spref)

    def graph (titles, Es, y, clusterCombos):
        pca = sklearn.decomposition.PCA(2)
        pX = pca.fit_transform(X)

        # Ground-truth
        fig, axs = plt.subplots(1, len(Es) + 1)
        fig.set_figheight(4)
        fig.set_figwidth(12)

        def plotArrows (ax, fro, to):
            for each in fro:
                ax.arrow(each[0], each[1], to[0] - each[0], to[1] - each[1], linewidth=0.5)

        for e in np.unique(y):  # ground-truth
            idxs = np.nonzero(y == e)[0]
            axs[0].scatter(pX[idxs,0], pX[idxs,1], marker='P' if e >= K else 'o')
            #plotArrows(axs[0], pX[idxs,:], pX[e,:])
        axs[0].set_title("Ground-truth")
        #axs[0].legend([ clusterCombos[e] for e in np.unique(y) ], fontsize='x-small')

        for i, E in enumerate(Es):
            for e in np.unique(E):  # inference
                idxs = np.nonzero(E == e)[0]
                axs[i+1].scatter(pX[idxs,0], pX[idxs,1], marker='P' if e >= N else 'o')
                #plotArrows(axs[i+1], pX[idxs], pX[e,:])
            axs[i+1].set_title(titles[i])
            #axs[i+1].legend([ combos[e] for e in np.unique(E) ], fontsize='x-small')
        #pp = PdfPages("mnist_clusters.pdf")
        #plt.savefig(pp, format='pdf')
        #pp.close()
        # plt.show()
        plt.savefig('plot.jpg')

    
    if method == 'CAP':
        s, E = fullCAP(S, x, Spref, y, clusterCombos)
    elif method == 'CAPsub':
        # s, E = subsetCAP(originalS, x, Spref, y, clusterCombos, min(80, N))
        someS = np.load(someSPath)
        somex = np.load(somexPath)
        idxs = np.load(idxsPath)
        s, E = subsetCAPOptimized(someS, somex, x, Spref, y, clusterCombos, 150, idxs)
    elif method == 'AP':
        s, E = APstandard(S, Spref, y, clusterCombos, 1)
    elif method == 'APheur':
        s, E = APstandard(S, Spref, y, clusterCombos, 1, True)
    elif method == 'AC':
        s, E = AC(S, x, y, clusterCombos, thresh)
    elif method == 'ACheur':
        s, E = AC(S, x, y, clusterCombos, thresh, True)
    elif method =='FCM':
        s, E = FuzzyCMeans(x, y, numClusters, clusterCombos, m)
    elif method =='GCA':
        someemb = np.load(embPath)
        s, E = GCA(y, someemb, numClusters, clusterCombos, m, spref2, g_path)
    elif method =='GMM':
        someemb = np.load(embPath)
        s, E = GMM(y, someemb, numClusters, clusterCombos, m, reg_covar=0.1)
    elif method == 'CKM':
        embs = np.load(embPath)
        s, E = CKM(y, embs, numClusters, clusterCombos, g_path)
    elif method == 'GCA_CKM':
        embs = np.load(embPath)
        s, E = GCA_CKM(y, embs, numClusters, clusterCombos, spref2)
    
    ari = adjusted_rand_score(E, y)

    if False:  # making a figure
        graph(["CAP", "AP+Heur"], [E1, E4 ], y, clusterCombos)
    return s, ari
