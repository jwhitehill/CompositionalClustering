import prepScore
import numpy as np
import sklearn.cluster
import sklearn.decomposition
import scipy.optimize
import itertools

def score (E, y, combos, clusterCombos):
    comparisons, comparisonsHat = prepScore.prepScore(E, y, combos, clusterCombos)
    #comparisons = []
    #comparisonsHat = []
    #for i in range(len(y)):
    #    for j in range(len(y)):
    #        if i != j:
    #            comparisons.append(set(clusterCombos[int(y[i])]).issubset(clusterCombos[int(y[j])]))
    #            comparisonsHat.append(set(combos[int(E[i])]).issubset(combos[int(E[j])]))
    comparisons = np.array(comparisons)
    comparisonsHat = np.array(comparisonsHat)
    return np.mean(comparisons == comparisonsHat)

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

# def AC (S, x, y, clusterCombos, thresh, useHeuristic = False):
#     print("AC---------")
#     N = len(x)
#     combos, comboInvMap = makeCombos(N)  # combos of datapoints
#     ag = sklearn.cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=thresh)
#     ag.fit(x)
#     clusterIndices = {}
#     for label in np.unique(ag.labels_):
#         idxs = np.nonzero(ag.labels_ == label)[0]
#         clusterIndices[label] = idxs[centroid(S, idxs)]
#     if useHeuristic and len(np.unique(ag.labels_)) < 8:
#         E = baselineMethod(ag.labels_, clusterIndices, y, S, combos, comboInvMap, clusterCombos)
#     else:
#         E = ag.labels_

#     theScore = score(E, y, combos, clusterCombos)
#     print("AC best: {}".format(theScore))
#     return theScore, E
