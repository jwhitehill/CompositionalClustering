def prepScore (E, y, combos, clusterCombos):
    comparisons = []
    comparisonsHat = []
    for i in range(len(y)):
        for j in range(len(y)):
            if i != j:
                comparisons.append(set(clusterCombos[y[i]]).issubset(clusterCombos[y[j]]))
                comparisonsHat.append(set(combos[E[i]]).issubset(combos[E[j]]))
    return comparisons, comparisonsHat

