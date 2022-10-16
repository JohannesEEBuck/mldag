#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 4 11:36:26 2021

@author: ngoc

Utility functions. 
"""


import numpy as np
import networkx as nx
import os
import pickle
import random
import matplotlib.pyplot as plt

#naming convention files
def _appendModelInfo(fname,q,smallR):
  return fname+"_q_"+ str(q) + "_r_" + str(smallR) + ".pk"

def saveTo(obj,save_folder,fname): 
  """ Save obj to save_folder/fname with pickle
  Create save_folder if it doesn't exist yet."""
  if not os.path.exists(save_folder):    
    os.makedirs(save_folder)
  with open(os.path.join(save_folder, fname), 'wb') as file:      
    pickle.dump(obj, file)   


def getReachability(A):
  """Returns the reachability matrix from the adjacency matrix A. 
  Param: 
    A: (d,d) np.array
  """
  #Amat = reshapeToMatrix(A)  
  G = nx.DiGraph(A)
  length = dict(nx.all_pairs_shortest_path_length(G))
  R = np.array([[length.get(m, {}).get(n, 0) > 0 for m in G.nodes] for n in G.nodes], dtype=np.int32)
  return R.T  
    
def getSupport(G):
  """Returns the support of a networkx graph as an np array """    
  B = nx.to_numpy_array(G)
  return(B != 0)

def getRiver(labels,river='lower-colorado',prefix=None):
  """Returns the support of the true river as an np array from given labels"""
  B = np.zeros((len(labels),len(labels)))
  label_inv = dict([(labels[i],i) for i in range(len(labels))])  
  edges = np.loadtxt(prefix+'data/' + river +  '/adjacency.txt', delimiter=' ',dtype=np.str)
  for e in edges:
    i,j = e
    B[label_inv[i],label_inv[j]] = 1
  return B    

def spanTree(W,isMin=True):
  """Wrapper to find min root-directed spanning tree in a graph with adjacency matrix W
  if isMin = False then find the max spanning tree instead
  """
  if isMin:
    G = nx.DiGraph(W.T*1.0)
  else:
    G = nx.DiGraph(W.T*-1.0)
  try:
    tree = nx.minimum_spanning_arborescence(G)
  except:
    #try returning the spanning arborescence anyway, without the is_arboresence test. 
    import networkx.algorithms.tree.branchings as bb
    ed = bb.Edmonds(G)
    tree = ed.find_optimum('weight', default=1, kind="min", style="arborescence", preserve_attrs=False)    
  return tree.reverse(copy=False)


def _count_accuracy(G_true, G_est, name_suffix = ''):
    """Compute various accuracy metrics for G_est.
    
    This code is a small modification of the count_accuracy function in
      https://github.com/xunzheng/notears/blob/master/notears/utils.py 
    Unlike Xunzheng et al, we use normalized SHD instead of SHD (see definition below)   

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        G_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        G_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG
        score_suffix: str
          string to append to each key name in the return matrix
          eg: str = '_r' would return keys 'fdr_r' instead of 'fdr'

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: (undirected extra + undirected missing + reverse) / (worst case = number of edges in true graph + number of edges in the estimated graph)
    """
    if not ((G_est == 0) | (G_est == 1)).all():
      raise ValueError('G_est should take value in {0,1}')
    d = G_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(G_est == -1)
    pred = np.flatnonzero(G_est == 1)
    cond = np.flatnonzero(G_true)
    cond_reversed = np.flatnonzero(G_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=False)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size =  d * (d - 1) - len(cond) #before it was 0.5*d*(d-1)-len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(G_est + G_est.T))
    cond_lower = np.flatnonzero(np.tril(G_true + G_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    #normalize shd
    shd = shd / (np.sum(G_true) + np.sum(G_est))
    return {'fdr'+name_suffix: fdr, 'tpr'+name_suffix: tpr, 'fpr'+name_suffix: fpr, 'shd'+name_suffix: shd}

def count_accuracy(G_true, G_list): 
  """Compute the accuracy scores of G_true vs each tree in G_list, 
  for both the tree *and* the reachability graphs. 
  
  Args: 
    G_true (np.ndarray): [d, d] ground truth graph, {0, 1}
    G_list: list
      each entry is an (np.ndarray): [d, d] estimated graph, {0, 1}. 

  Returns: 
    list of scores
    each entry is a dictionary with keys
    ['fdr', 'tpr', 'fpr', 'shd', 'fdr_r', 'tpr_r', 'fpr_r', 'shd_r']
    
  """
  scores_dict = []
  for G_est in G_list: 
    scores = _count_accuracy(G_true,G_est)  
    scores_dict += [scores]
  return scores_dict



def create_sa(d,c1,c2):
    
    """Generates a random spanning arborescence with d nodes and edge weights uniformly between c1 and c2
    
    Args: 
      d (int): Number of nodes
      c1 (float): minimum edge weight
      c2 (float): maximum edge weight
    
    Returns: 
      C (d x d numpy array): Edge weight matrix
      
    """
    
    
    G=nx.generators.trees.random_tree(d)
    
    v=[random.choice(list(G.nodes))]
    
    A=np.zeros((d,d))
    
    while len(v):
        v2=[]
        for i in v:
            n=list(G.neighbors(i))
            for n_k in n:
                G.remove_edge(n_k,i)
            A[n,i]=1
            v2+=n
            
        v=np.copy(v2)
        
    C=np.multiply(np.random.uniform(c1, high=c2, size=(d,d)),A)           
    
    
    
    return(C)
    


def fromCtoB(C):
    
    """Calculates the Kleene Star Matrix B from C (Max-times)
    
    Args: 
      C (d x d numpy array): Edge weight matrix
    
    Returns: 
      B (d x d numpy array): Kleene star matrix
      
    """

    
    d=np.size(C,0)
    
    G=nx.DiGraph(C)
    top_sort=list(nx.topological_sort(G))
    
    B=np.copy(C)+np.eye(d)
    
    #different distance in top. order
    for idx in range(2,d):        
        for idy, val in enumerate(top_sort[0:d-idx]):
            B[val,top_sort[idy+idx]]=np.amax(np.multiply(B[val,top_sort[idy:(idy+idx)]],B[top_sort[idy:(idy+idx)],top_sort[idx+idy]]))
              
    return(B)



def MLMatrixMult(Z,B):
    
    """Calculates max-linear data, returns log-data
    
    Args: 
      Z (n x d numpy array): Array of all innovations
    
    Returns: 
      X (n x d numpy array): max-linear log-data
      
    """
    

    n=np.size(Z,0)
    d=np.size(Z,1)
    
    X=np.zeros((n,d))
    
    for i in range(n):
        Y=[np.multiply(Z[i,:],B[:,j]) for j in range(d)]
        X[i,:]=np.amax(Y, axis=1)
        
    return(X)   



def MLMatrixMult2(Z,B):
    
    """Calculates max-linear data, returns log-data
    
    Args: 
      Z (n x d numpy array): Array of all innovations
    
    Returns: 
      X (n x d numpy array): max-linear log-data
      
    """
    W=np.copy(B)

    W[W>0]=1

    B=np.log(B)
    np.fill_diagonal(B, 0)    

    n=np.size(Z,0)
    d=np.size(Z,1)
    
    X=np.zeros((n,d))
    
    for i in range(n):
        Y=[np.multiply(Z[i,:],W[:,j])+B[:,j] for j in range(d)]
        X[i,:]=np.amax(Y, axis=1)
        
    return(X)  
# 

def updateR(R,j,i):
    
    """This function updates the Reachability Matrix R if an edge from j to i is added
    
    Args: 
      R (d x d numpy array): Reachability matrix
      j (int): index with j<d
      i (int): index with i<d
    
    Returns: 
      R (d x d numpy array): Updated Reachability matrix
      
    """
    
    an=list(np.where(R[:,j]==1)[0])+[j]
    su=list(np.where(R[i,:]==1)[0])+[i]
    
    con=[(x,y) for x in an for y in su]
    
    for k in con:
        R[k]=1

    return(R)

def plotParameterSelection(dataset,B, B_est_coll, lambdarange,alpharange):
    
"""
Recreate the Plot for the Parameter Selection

Args:
    dataset (str): name of the dataset 
    B (numpy dxd array): True Kleene star matrix of the dataset
    B_est_coll(dict of list of d x d numpy arrays): Output is a dictionary. For each key (lambda,alpha),  
        B_est_coll contains a list of estimated Kleene star matrices for the original dataset and its subsamples
    lambda_range (list of floats in [0,1]): All values of lambda
    alpha_range (list of floats in [0,1]): All values of alpha
    

Returns:
     Saves the plots as in "Learning Bayesian Networks from Extreme Data" (cf. Figures 3-7) 
     in the subfolder plots_parameterselection/dataset.png
"""
    
    plt.ioff()
    
    n_range=len(B_est_coll[(lambdarange[0],alpharange[0])])-1
    shd=[]
    tpr=[]
    shdcoll=np.zeros((len(lambdarange), n_range))
    tprcoll=np.zeros((len(lambdarange), n_range))
    
    param_best={"alpha": np.inf, "lambda": np.inf}
    
    shd_best=np.inf
    
    
    
    for idx, lambda1 in enumerate(lambdarange):
        for alpha in alpharange:
            
            B_est=B_est_coll[(lambda1, alpha)][0]
            
            scores = count_accuracy(B_est_coll[(lambda1,alpha)][0],B_est_coll[(lambda1,alpha)][1:])
            shdcoll[idx,:]=[b["shd"] for b in scores]
            tprcoll[idx,:]=[b["tpr"] for b in scores]
            
            shd+=[count_accuracy(B,[B_est])[0]["shd"]]
            tpr+=[count_accuracy(B,[B_est])[0]["tpr"]]
    
    
            if np.mean([b["shd"] for b in scores])<shd_best:
    
               shd_best=np.mean([b["shd"] for b in scores])
               param_best['alpha']=alpha
               param_best['lambda']=lambda1
               
    if len(lambdarange)==1:
        x=alpharange.copy()
        xlabelname=r"$\alpha$"
    else:
        x=lambdarange.copy()
        xlabelname=r"$\lambda$"

    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, shd, 'g-', marker="o",linewidth=2)
    ax2.plot(x, tpr, 'b-', marker="o",linewidth=2)
    ax1.boxplot(np.transpose(shdcoll), positions=x, widths=0.03)
    ax1.axvline(x=param_best["lambda"], color="r")
    plt.xlim([min(x)-0.03, max(x)+0.03])
    plt.grid(axis = 'y')
    ax1.set_xlabel(xlabelname,size=18)
    ax1.set_ylabel('nSHD',size=18)
    ax2.set_ylabel('TPR',size=18)
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    plt.title("nSHD and TPR for "+ dataset,size=20)
    ax2.set_ylim([0,1.1])
    ax1.set_ylim([0,0.8])
    
    if not os.path.exists("plots_parameterselection"):    
        os.makedirs("plots_parameterselection")
  
    name="plots_parameterselection/"+dataset+".png"
    plt.savefig(name)
