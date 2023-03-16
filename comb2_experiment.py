import numpy as np
from multiprocessing import Pool
from experiments import experiments
import json

def getMeanAndSE(res):
    n = len(res)
    res = np.array(res)
    mean = np.mean(res)
    std = np.sqrt(sum((res-mean)**2)/(n-1))
    return mean, std/np.sqrt(n)

def printAndLog(prefix, path, results, method):
    with open(path, 'a') as f:
        f.write(json.dumps(results)+'\n')
        mean, se = getMeanAndSE([result[0] for result in results])
        log = f'{prefix} {n*15}, method {method} CRI: mean {mean} se {se}'
        print(log)
        f.write(log+'\n')
        mean, se = getMeanAndSE([result[1] for result in results])
        log = f'{prefix} {n*15}, method {method} ARI: mean {mean} se {se}'
        print(log)
        f.write(log+'\n')

if __name__ == '__main__':
    '''
    hyperparameters are searched in the previous step, the best hyperparameters are hard coded here
    '''
    # Librispeech #
    dirPath = 'data'
    dataset = 'librispeech_5'
    savepath = 'comb2_librispeech.log'
    for n in [10, 50, 100]:
        xPaths = [f'{dirPath}/{dataset}_{n*15}/{trial}_x.npy' for trial in range(10)]
        XPaths = [None for _ in range(10)]
        SPaths = [f'{dirPath}/{dataset}_{n*15}/{trial}_SShort.npy' for trial in range(10)]
        embPaths = [f'{dirPath}/{dataset}_{15*n}/{trial}_emb.npy' for trial in range(10)]

        # AP #
        results = []
        for i in range(10):
            result = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 1, 1, 'AP', None, None, None, None)
            results.append(result)
        printAndLog('Librispeech', savepath, results, 'AP')

        # AC #
        results = []
        for i in range(10):
            result = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 1, 1, 'AC', None, None, None, None)
            results.append(result)
        printAndLog('Librispeech', savepath, results, 'AC')

        # FCM #
        results = []
        for i in range(10):
            result = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 1, 1, 'FCM', None, None, None, None, 18, 1.2)
            results.append(result)
        printAndLog('Librispeech', savepath, results, 'FCM')

        # CAP #
        someSPath = [f'{dirPath}/{dataset}_{n*15}/{trial}_someS.npy' for trial in range(10)]
        somexPath = [f'{dirPath}/{dataset}_{n*15}/{trial}_somex.npy' for trial in range(10)]
        idxsPath = [f'{dirPath}/{dataset}_{n*15}/{trial}_idxs.npy' for trial in range(10)]
        results = []
        for i in range(10):
            result = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 4, 1, 'CAPsub', None, someSPath[i], somexPath[i], idxsPath[i])
            results.append(result)
        printAndLog('Librispeech', savepath, results, 'CAP')

        # GMM #
        results = []
        for i in range(10):
            result = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 4, 1, 'GMM', embPaths[i], None, None, None, 18, 0.55, 0.05)
            results.append(result)
        printAndLog('Librispeech', savepath, results, 'GMM')

        # GCA #
        results = []
        for i in range (10):
            s = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 2, 0, 'GCA', embPaths[i], None, None, None, numClusters=17, spref2=0.4, g_path='checkpoints/librispeech.pt')
            results.append(s)
        printAndLog('Librispeech', savepath, results, 'GCA')
        

        # CKM #
        results = []
        for i in range(10):
            s = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 2, 0, 'CKM', embPaths[i], None, None, None, numClusters=6, g_path='checkpoints/librispeech.pt')
            results.append(s)
        printAndLog('Librispeech', savepath, results, 'CKM')


    
    # OmniGlot #
    dirPath = 'data'
    dataset = 'CAP_omniglot_data_5'
    savepath = 'comb2_omniglot.log'
    for n in [10, 50, 100]:
        xPaths = [f'{dirPath}/{dataset}_{n*15}/{trial}_x.npy' for trial in range(10)]
        XPaths = [None for _ in range(10)]
        SPaths = [f'{dirPath}/{dataset}_{n*15}/{trial}_SShort.npy' for trial in range(10)]
        embPaths = [f'{dirPath}/{dataset}_{15*n}/{trial}_emb.npy' for trial in range(10)]

        # AP #
        results = []
        for i in range (10):
            s = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 1, 1, 'AP', None, None, None, None)
            results.append(s)
        printAndLog('OmniGlot', savepath, results, 'AP')

        # AC #
        results = []
        for i in range (10):
            s = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 1, 1, 'AC', None, None, None, None)
            results.append(s)
        printAndLog('OmniGlot', savepath, results, 'AC')

        # FCM #
        results = []
        for i in range (10):
            s = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 1, 1, 'FCM', None, None, None, None, 18, 1.2)
            results.append(s)
        printAndLog('OmniGlot', savepath, results, 'FCM')

        # GMM #
        results = []
        for i in range (10):
            s = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 4, 1, 'GMM', embPaths[i], None, None, None, 17, 0.4, 0.05)
            results.append(s)
        printAndLog('OmniGlot', savepath, results, 'GMM')

        # CAP #
        someSPath = [f'{dirPath}/{dataset}_{n*15}/{trial}_someS.npy' for trial in range(10)]
        somexPath = [f'{dirPath}/{dataset}_{n*15}/{trial}_somex.npy' for trial in range(10)]
        idxsPath = [f'{dirPath}/{dataset}_{n*15}/{trial}_idxs.npy' for trial in range(10)]
        results = []
        for i in range (10):
            s = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 2, 1, 'CAPsub', None, someSPath[i], somexPath[i], idxsPath[i])
            results.append(s)
        printAndLog('OmniGlot', savepath, results, 'CAPsub')

        # GCA #
        results = []
        for i in range (10):
            s = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 2, 0, 'GCA', embPaths[i], None, None, None, numClusters=17, spref2=0.4, g_path='checkpoints/omniglot.pt')
            results.append(s)

        printAndLog('OmniGlot', savepath, results, 'GCA')
    
        # CKM #
        results = []
        for i in range(10):
            s = experiments(5, n, xPaths[i], XPaths[i], SPaths[i], 2, 0, 'CKM', embPaths[i], None, None, None, numClusters=5, g_path='checkpoints/omniglot.pt')
            results.append(s)
        printAndLog('OmniGlot', savepath, results, 'CKM')