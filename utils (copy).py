import numpy as np
import sys
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
import scipy as sci
from sklearn.cluster import KMeans
import sklearn.metrics as sm

# Compute the similarity graph on nodes
# Return the resulted adjacency after tunning by sigma and epsilon
def epsilon_similarity_graph(X: np.ndarray, sigma=1, epsilon=0):
    distance=squareform(pdist(X, 'euclidean'))
    weights=np.exp(-distance**2/(2*sigma*sigma))
    np.fill_diagonal(weights,0)
    adjacency=weights
    adjacency[adjacency<epsilon]=0
    return adjacency

   
# Compute laplacian matric of the adjacency matrix
# Combinatorial version or normalized version can be control by "normalize"
def compute_laplacian(adjacency: np.ndarray, normalize: bool):
    D = np.diag(np.sum(adjacency, 1)) # Degree matrix
    combinatorial = D - adjacency
    if normalize:
        D_norm = np.diag(np.clip(np.sum(adjacency, 1), 1, None)**(-1/2))
        return D_norm @ combinatorial @ D_norm
    else:
        return combinatorial

# Compute the spectral decomposition of laplacian matrix
def spectral_decomposition(laplacian: np.ndarray):
    lamb, U=sci.linalg.eigh(laplacian)
    sorted_idx = np.argsort(lamb)
    lamb = lamb[sorted_idx]
    U=U[:,sorted_idx]
    return lamb,U

# Compute the GFT of the signal
def GFT(signal, U):
    return U.T @ signal
# Compute the inverse of GFT of the signal
def iGFT(fourier_coefficients, U):
    return U @ fourier_coefficients

# Apply GFT to x and filter x based on the spectral response
# Finaly convert the result back to graph domain
def ideal_graph_filter(x, spectral_response, U):
    return iGFT(GFT(x,U) * spectral_response, U)

# Evaluation the prediction using Fourier analysis
def pred_iteration(A,iters, y, n, filtered_x_lp):
    f1_scores =[]
    y_ = y.copy() # this is training data
    for i in range(iters):
        # choose randomly n indices to masking, for evaluating use
        test_idx = np.random.choice(np.arange(len(y_)),n,replace = False)
        # masking some winner
        y_[test_idx]=0
        # prepare ground truth labels
        truth = (y[test_idx]).values
        # prepare for the prediction
        pred = []
        for i in test_idx:
            l = np.where(A[i] !=0)[0]  # searching neigbhours for a masked node
            if(len(l)!= 0):
                tmp = 0 # filtered_x_lp[i] => add initial node value ? or without mean
                for j in l:
                    # sum over values from neighbour nodes
                    tmp += filtered_x_lp[j]
                # compute mean according to total number of neighbours
                pred.append(tmp/len(l))   
            else:
                # if the node has no neighbour then the value will be its signal values
                pred.append(filtered_x_lp[i])

        # thresholding over the prediction so that only 1 or -1 will be returned
        pred_thres = np.array(pred)
        pred_thres [pred_thres >0 ] = 1
        pred_thres [pred_thres <0 ] = -1

        # compute the f1 score of the prediction and add to scores list
        f1_scores.append(sm.f1_score(truth,pred_thres))
        
    # compute mean of all obtained scores
    mean = np.mean(f1_scores)
    # compute variance of all obtained scores
    var = np.std(f1_scores)
    print("The mean is ",mean)
    print("The variance is ",var)
    return mean,var


#############################
### GCN PART ################

import time

import networkx as nx
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl import DGLGraph
from dgl.data.citation_graph import load_cora

np.random.seed(0)
torch.manual_seed(1)

# Define LaplacianPolynomial class
class LaplacianPolynomial(nn.Module):
    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 k: int,
                 dropout_prob: float,
                 norm=True):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k
        self._norm = norm
        # Contains the weights learned by the Laplacian polynomial
        self.pol_weights = nn.Parameter(torch.Tensor(self._k + 1))
        # Contains the weights learned by the logistic regression (without bias)
        self.logr_weights = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        torch.manual_seed(0)
        torch.nn.init.xavier_uniform_(self.logr_weights, gain=0.01)
        torch.nn.init.normal_(self.pol_weights, mean=0.0, std=1e-3)

    def forward(self, graph, feat):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.

        Parameters
        ----------
        graph (DGLGraph) : The graph.
        feat (torch.Tensor): The input feature

        Returns
        -------
        (torch.Tensor) The output feature
        """
        feat = self.dropout(feat)
        graph = graph.local_var()
        
        # D^(-1/2)
        norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp)

        # mult W first to reduce the feature size for aggregation.
        feat = torch.matmul(feat, self.logr_weights)

        result = self.pol_weights[0] * feat.clone()

        for i in range(1, self._k + 1):
            old_feat = feat.clone()
            if self._norm:
                feat = feat * norm
            graph.ndata['h'] = feat
            # Feat is not modified in place
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            if self._norm:
                graph.ndata['h'] = graph.ndata['h'] * norm

            feat = old_feat - graph.ndata['h']
            result += self.pol_weights[i] * feat

        return result

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        return summary.format(**self.__dict__)

# Train the model
def train(model, g, features, labels, loss_fcn, train_mask, optimizer):
    model.train()  # Activate dropout
    
    logits = model(g, features)
    loss = loss_fcn(logits[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
#Evaluate the model
def evaluate(model, g, features, labels, mask):
    model.eval()  # Deactivate dropout
    with torch.no_grad():
        logits = model(g, features)[mask]  # only compute the evaluation set
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
# Compute the laplacian polynomial filter
def polynomial_graph_filter(coeff: np.array, laplacian: np.ndarray):
    res = np.zeros_like(laplacian)
    for i in range (len(coeff)):
        res += coeff[i] * np.linalg.matrix_power(laplacian, i)
    return res

# Compute the polynomial graph filter response
def polynomial_graph_filter_response(coeff: np.array, lam: np.ndarray):
    res = np.zeros((len(coeff),len(lam)))
    for i in range(len(coeff)):
        res[i] = coeff[i] * (lam**i)
    res = np.sum(res,axis =0)
    return res

# Apply Graph convolutional networks model to the graph
def apply_gcn(iters,X_,y_,A_,laplacian_,lamb,U_):
    # do some basic copies 
    X = X_.copy()
    y = y_.copy()
    A = A_.copy()
    U = U_.copy()
    laplacian = laplacian_.copy()

    # Some basic settings
    features = torch.FloatTensor(X)
    labels = torch.LongTensor(y) 
    in_feats = features.shape[1]  # 2
    n_classes = 2
    n_edges = int(A.sum() // 2)
    pol_order = 3
    lr = 0.2
    weight_decay = 5e-6
    n_epochs = 500
    p_dropout = 0.8
    
    f1_scores = []
    #print("Computing")
    # Start
    for i in range(iters):
        # display the processing level
        #if( i != 0 and i%(iters*0.1) == 0):
        #    print(str(int(i*100/iters))+" %")

        # prepare for masking
        n_points = X.shape[0]
        indices = np.arange(n_points)
        np.random.shuffle(indices)
        split_t = int(n_points*0.2)
        test_idx = indices[:split_t]
        train_idx = indices[split_t:]
        train_mask = np.zeros(n_points)
        train_mask[train_idx] = 1
        test_mask = np.zeros(n_points)
        test_mask[test_idx] = 1
        graph = nx.from_numpy_matrix(A)
        adjacency = np.asarray(nx.to_numpy_matrix(graph))

        # create the graph 
        graph = DGLGraph(graph)
        model = LaplacianPolynomial(in_feats, n_classes, pol_order, p_dropout)

        loss_fcn = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)
        # train the model
        for epoch in range(50):
            loss = train(model, graph, features, labels, loss_fcn, train_mask, optimizer)
        # get the gcn coefficients
        coeff_gcn =  model.pol_weights.detach().numpy()
        graph_gcn_filter = polynomial_graph_filter(coeff_gcn, laplacian)
        features_gcn = graph_gcn_filter @ features.numpy()
        train_mask = torch.BoolTensor(train_mask) 
        test_mask = torch.BoolTensor(test_mask)
        train_features_gcn = features_gcn[train_mask,:]
        train_labels = labels[train_mask]
        test_features_gcn = features_gcn[test_mask,:]
        test_labels = labels[test_mask]

        model =  LogisticRegression(C=1000,penalty = 'l2',solver='liblinear', multi_class='auto',max_iter = 2000)
        model.fit(train_features_gcn, train_labels)

        # compute predictions of gcn and evaluate the performance of model        
        test_pred = model.predict(test_features_gcn)
        f1_scores.append(sm.f1_score(test_labels,test_pred))

        
    #print("100 %")
    # compute mean of all obtained scores
    mean = np.mean(f1_scores)
    # compute variance of all obtained scores
    var = np.std(f1_scores)
    print("The mean of f1 score is ",mean)
    print("The variance of f1 score is ",var)
    return mean,var        
