import torch
import glob
import torch.nn.functional as F
import torch.nn as nn
from experiments import makeCombos3, AC, APstandard, FuzzyCMeans3, GMM3
import numpy as np
import itertools
import sklearn.cluster
from score_use import score
from torch import optim
import random
from sklearn.metrics.cluster import adjusted_rand_score
from multiprocessing import Pool
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 32

def getMeanAndSE(res):
    n = len(res)
    res = np.array(res)
    mean = np.mean(res)
    std = np.sqrt(sum((res-mean)**2)/(n-1))
    return mean, std/np.sqrt(n)

class DistLoss(nn.Module):
    def __init__(self, init):
        super(DistLoss, self).__init__()
        self.W = nn.Parameter(torch.tensor(init))
        self.clusterCombos, _ = makeCombos3(len(init))
        self.g_net = G_net().to(device)
        self.g_net.load_state_dict(torch.load("checkpoints/comb3_g.pt"))

    def forward(self, x):
        numClusters = self.W.size(0)
        embComp2s = []
        embComp3s = []

        for i in range(numClusters-1):
            for j in range(i + 1, numClusters):
                embComp2 = g_net(self.W[i].unsqueeze(0), self.W[j].unsqueeze(0))
                embComp2s.append(embComp2)

        for i in range(numClusters-2):
            for j in range(i + 1, numClusters-1):
                embComp2Temp = g_net(self.W[i].unsqueeze(0), self.W[j].unsqueeze(0))
                for k in range(j + 1, numClusters):
                    embComp3 = g_net(embComp2Temp, self.W[k].unsqueeze(0))
                    embComp3s.append(embComp3)
        embComp2s = torch.cat(embComp2s, 0)
        embComp3s = torch.cat(embComp3s, 0)

        allCentroids = torch.cat([self.W, embComp2s, embComp3s], 0)

        assignment = torch.argmin(torch.cdist(F.normalize(x), F.normalize(allCentroids)), 1)
        assignCentroids = allCentroids[assignment]
        dists = torch.dist(F.normalize(assignCentroids), F.normalize(x))
        return dists, assignment

class G_net (nn.Module):
    def __init__ (self):
        super(G_net, self).__init__()
        self.linear1a = nn.Linear(EMBEDDING_DIM, 32)
        self.linear1b = nn.Linear(EMBEDDING_DIM, 32)

    def forward (self, X1, X2):
        linear1 = self.linear1a(X1) + self.linear1a(X2) + self.linear1b(X1*X2)
        return linear1

class Lstm(nn.Module):
    def __init__(self, model_dim=30):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(model_dim, 256, 2, batch_first = True)
        self.outlayer = nn.Linear(256, 32)
        
        
    def forward(self, inputs, mask=None):
        out, _ = self.lstm(inputs)
        out = out[:,-1, :]
        out = self.outlayer(F.leaky_relu(out))
        return out
    
def GCA3(y, clusterCombos, x, someemb, numClusters, spref2, g_net):
    ag = sklearn.cluster.AgglomerativeClustering(n_clusters=numClusters, distance_threshold=None)
    ag.fit(x)

    E = ag.labels_
    numClusters = max(E)+1
    centroids = [[] for _ in range(numClusters)]
    for e_cnt, e in enumerate(E):
        centroids[e].append(someemb[e_cnt])
    cs = []
    for c in centroids:
        cs.append(torch.mean(torch.stack(c), 0))
    cs = torch.stack(cs).to(device)

    embComp2s = []
    embComp3s = []
    clusterCombosMap, invClusterCombosMap = makeCombos3(numClusters)
    combIdx = clusterCombosMap[numClusters:]
    with torch.no_grad():
        for i in range(numClusters-1):
            for j in range(i + 1, numClusters):
                embComp2 = g_net(cs[i].unsqueeze(0), cs[j].unsqueeze(0))
                embComp2s.append(embComp2)

        for i in range(numClusters-2):
            for j in range(i + 1, numClusters-1):
                embComp2Temp = g_net(cs[i].unsqueeze(0), cs[j].unsqueeze(0))
                for k in range(j + 1, numClusters):
                    embComp3 = g_net(embComp2Temp, cs[k].unsqueeze(0))
                    embComp3s.append(embComp3)
    embComp2s = torch.cat(embComp2s, 0)
    embComp3s = torch.cat(embComp3s, 0)

    embComp = torch.cat([embComp2s, embComp3s], 0)
    embcombNorm = F.normalize(embComp).data.cpu().numpy()
    embNorm = F.normalize(cs).data.cpu().numpy()
    
    S = np.zeros((len(embNorm), len(embcombNorm)))

    for xCnt in range(len(embNorm)):
        S[xCnt] = np.linalg.norm(embNorm[xCnt, :] - embcombNorm, axis=1) **2

    for cCnt, c in enumerate(combIdx):
        for singleC in c:
            S[singleC, cCnt] = np.inf


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
    
    finalE = []
    for e in E:
        finalE.append(newE[e])
    theScore = score(finalE, y, clusterCombosMap, clusterCombos)
    return theScore, finalE

def CKM3(y, x, numClusters, clusterCombos, g_net):
    
    min_loss = 100
    for init_cnt in tqdm(range(100)):
        # initCentroids = x[np.array([0, 10,20,30, 40])]
        # initCentroids = np.load('CKM_best_init.npy')
        # random select initial points
        initCentroids = x[random.sample(range(len(x)), numClusters)]
        # create main logic
        computeLoss = DistLoss(initCentroids).to(device)

        optimizer = optim.Adam([computeLoss.W], lr=0.2)
        xTorch = torch.from_numpy(x).to(device)

        # load pre saved checkpoint
        # computeLoss.load_state_dict(torch.load('saved_model.pt'))
        for iterCnt in range(100):
            optimizer.zero_grad()
            loss, assignment = computeLoss(xTorch) # .forward
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            if loss < min_loss:
                min_loss = loss.detach()
                E = assignment.cpu().data.numpy()
        # print(init_cnt, loss)
    clusterCombosPred, _ = makeCombos3(numClusters)
    theScore = score(E, y, clusterCombosPred, clusterCombos)
            #     print(init_cnt, loss, theScore)

            # save checkpoint
            # torch.save(computeLoss.state_dict(), 'saved_model.pt')
    return theScore, E

def testMethod(method, dataidx, numCluster=28):
    data = torch.load(files[dataidx]).float().to(device).permute(1, 0, 2, 3).contiguous().view(-1, 199, 32)
    datas = []
    batch_cnt = data.size(0)//50
    for i in range(batch_cnt):
        subx = f_net(data[i*50:i*50+50])
        datas.append(subx)
    x = torch.cat(datas, 0)
    x_norm = F.normalize(x)
    s = torch.matmul(x_norm, x_norm.transpose(0, 1))
    x_numpy = x_norm.data.cpu().numpy()
    s_numpy = s.data.cpu().numpy()
    
    if method == 'ckm3':
        evalscore, E = CKM3(y, x.data.cpu().numpy(), 6, clusterCombosMap, g_net)
        ari = adjusted_rand_score(E, y)
        # print(evalscore)
    elif method == 'gca3':
        evalscore, E = GCA3(y, clusterCombosMap, x_numpy, x, 28, 0.4, g_net)
        ari = adjusted_rand_score(E, y)
        # print(evalscore)
    elif method == 'AC':
        evalscore, E = AC(s_numpy, x_numpy, y, clusterCombos=clusterCombosMap, thresh=1)
        ari = adjusted_rand_score(E, y)
        # print(evalscore)
    elif method == 'AP':
        evalscore, E = APstandard(s_numpy, pref=1, y=y, clusterCombos=clusterCombosMap, discount=1)
        ari = adjusted_rand_score(E, y)
        # print(evalscore)
    elif method == 'FCM3':
        evalscore, E = FuzzyCMeans3(x_numpy, y, 28, clusterCombosMap, 1.2)
        ari = adjusted_rand_score(E, y)
        # print(evalscore)
    elif method == 'gmm3':
        evalscore, E = GMM3(y, x.data.cpu().numpy(), 28, clusterCombosMap, 0.55, 0.05)
        ari = adjusted_rand_score(E, y)
        # print(evalscore)
    return evalscore, ari



for datacnt in [10, 50, 100]:

    clusterCombosMap, invClusterCombosMap = makeCombos3(5)

    f_net = Lstm(32).to(device)
    g_net = G_net().to(device)
    f_net.load_state_dict(torch.load("checkpoints/comb3_f.pt"))
    g_net.load_state_dict(torch.load("checkpoints/comb3_g.pt"))
    files = glob.glob(f'comb3_cap_data/{datacnt}/*.pt')
    y = np.array([i for i in range(25) for _ in range(datacnt)])



    for method in ['AC', 'AP', 'FCM3', 'gmm3', 'gca3']:
        results = []
        for i in range(10):
            res = testMethod(method, i)
            results.append(res)
        with open('comb3.log', 'a') as f:
            f.write(json.dumps(results)+'\n')
        mean, se = getMeanAndSE([r[0] for r in results])
        log = f'k = {datacnt}, method {method} CRI: mean {mean} se {se}'
        print(log)
        with open('comb3.log', 'a') as f:
            f.write(log+'\n')
        mean, se = getMeanAndSE([r[1] for r in results])
        log = f'k = {datacnt}, method {method} ARI: mean {mean} se {se}'
        print(log)
        with open('comb3.log', 'a') as f:
            f.write(log+'\n')