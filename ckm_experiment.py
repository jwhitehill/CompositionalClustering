import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from itertools import combinations
from score_use import score, makeCombos
import numpy as np
from cap_evaluation import getMeanAndSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class G_net (nn.Module):
    def __init__ (self):
        super(G_net, self).__init__()
        self.linear1a = nn.Linear(32, 32)
        self.linear1b = nn.Linear(32, 32)
        # self.bn1 = nn.BatchNorm1d(32)

    def forward (self, X1, X2):
        linear1 = self.linear1a(X1) + self.linear1a(X2) + self.linear1b(X1*X2)
        return linear1

def pairwiseDists (A, B):
    A = A.reshape(A.shape[0], 1, A.shape[1])
    B = B.reshape(B.shape[0], 1, B.shape[1])
    D = A - B.transpose(0,1)
    return torch.norm(D, p=2, dim=2)

class DistLoss(nn.Module):
    def __init__(self, init):
        super(DistLoss, self).__init__()
        self.W = nn.Parameter(torch.tensor(init))
        self.combinations2 = torch.tensor(list(combinations(list(range(len(init))),2)))
        self.g_net = G_net().to(device)
        # self.g_net.load_state_dict(torch.load("omniglot_checkpoint/g_net_no_bn.pt"))
        self.g_net.load_state_dict(torch.load("/home/lizeqian/subset_embeddings_new_2/checkpoint_libri3_CAP/0.868_g.pt"))

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

def CKM(y, x, numClusters, clusterCombos):
    
    min_loss = 100
    for init_cnt in range(100):
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
        for iterCnt in range(1000):
            optimizer.zero_grad()
            loss, assignment = computeLoss(xTorch) # .forward
            loss.backward()
            optimizer.step()
            # print(iterCnt, loss)
        with torch.no_grad():
            if loss < min_loss:
                min_loss = loss.detach()
                E = assignment.cpu().data.numpy()
    clusterCombosPred, _ = makeCombos(numClusters)
    theScore = score(E, y, clusterCombosPred, clusterCombos)
            #     print(init_cnt, loss, theScore)

            # save checkpoint
            # torch.save(computeLoss.state_dict(), 'saved_model.pt')
    return theScore, E

if __name__ == '__main__':
    '''
    Librispeech Experiments
    '''
    K=5
    clusterCombos, _ = makeCombos(K)

    NSample = 10
    for numClusters in [5]:
        y = [i for i in range(len(clusterCombos)) for _ in range(NSample)]
        ss = []
        # numClusters = 6
        for i in range(10, 20):
            print(i)
            # embs = np.load(f'/home/lizeqian/data1t_hdd/CAP_omniglot_data_without_bn/CAP_omniglot_data_5_{15*NSample}/{i}_emb.npy')
            embs = np.load(f'/home/lizeqian/data1t_hdd/librispeech_5_{15*NSample}/{i}_emb.npy')
            s, E = CKM(y, embs, numClusters, clusterCombos)
            ss.append(s)
        print(ss)
        mean, se = getMeanAndSE(ss)
                
        print(f'Librispeech k = 5, n = {NSample*15}, c = {numClusters}, score: mean {mean} se {se}\n')

