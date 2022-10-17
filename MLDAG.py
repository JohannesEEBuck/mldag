#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 06 12:35:30 2022

@author: JohannesBuck

The estimator
for max-linear Bayesian networks (MLDAG) without automated parameter selection. 
(cf Algorithm 1 in "Learning Bayesian Networks from Extreme Data" from Buck and Tran)

"""


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from utils import create_sa, fromCtoB, MLMatrixMult, count_accuracy, _count_accuracy, saveTo
import os
import pickle
import networkx as nx
import sklearn
import sys
import scipy.cluster


class Network(nn.Module):
    
    """
    Defines the max-linear network in terms of a neural network with torch  
    """
    
    
    def __init__(self, dim):
        """
        Initialization of the network
        """
        super(Network, self).__init__()
        d=dim
        self.linears = nn.ModuleList([nn.Linear(1, d) for i in range(d)])
        self.dim = dim
    
    def forward(self, x, E): 
        """
        Forward function: For the current model, it calculates the max-linear matrix multiplication
        
        Args:
            x: (n x d torch array): Matrix of n d-dimensional observations
            E: (bool): If True, calculates C^(d-1)*x (cf Algorithm 1, line 4 in Learning Bayesian Networks 
                       from Extreme Data), if False, it calculates the  C*x; Here C are the weights of the neural
                       network
        Returns:
            The matrix multiplication explained above
                    
        """
        
        d = self.dim
        n= x[0].size()[0]
        X2=x[1]
        
        y=torch.zeros((d,d,n))
        
        if E==True:
            ite=d
        else:
            ite=1

    
        for i in range(ite):
                
            for i, l in enumerate(self.linears):
    
                y[i,:,:] = torch.transpose(l(x[0][:,i].view(-1,1)),0,1)#-torch.transpose(C_norm[i].repeat(n,d),0,1)
    
            
            x[0]=torch.cat((torch.transpose(x[0],0,1).view(-1,d,n), y), 0)
            x[0]=torch.transpose(torch.max(x[0], axis=0).values,0,1)
        x[0]=torch.cat((torch.transpose(x[0],0,1).view(-1,d,n), torch.transpose(x[1],0,1).view(-1,d,n)), 0)
        output=torch.transpose(torch.max(x[0], axis=0).values,0,1)
        
        return output

    
    def h_func(self):
        """
        Calculates the h value according to the Paper "DAGs with NO TEARS: Continuous Optimization for Structure Learning"
        by Zheng et al
        
        Returns:
            h (torch float): The h value (which is zero if and only if the underlying network is a DAG)
                    
        """
        d = self.dim
        
        W=torch.exp(self.linears[0].bias).view(d,-1)
        
        for i in range(1,len(self.linears)):
          W=torch.cat((W,torch.exp(self.linears[i].bias.view(d,-1))),1)
            
        W.fill_diagonal_(0)
            
        E = torch.matrix_exp(torch.mul(W,W))  # (Zheng et al. 2018)
        h = torch.trace(E) - d
        
        return h
    
    

    
    def weight_constraint(self):  
        """
        Calculates the regularization term f of the model as in equation (15) of "Learning Bayesian Networks 
                       from Extreme Data"
        
        Returns:
             (torch float): The regularization term 
                    
        """
        reg=0
        
        for i, l in enumerate(self.linears):
                reg+=torch.sum(torch.exp(l.bias))
                 
        return (-torch.log(reg))

    
    @torch.no_grad()
    def layer_to_adj(self) -> np.ndarray:
        """
        Converts the weights of the Neural Network to a Kleene star matrix
        
        Returns:
             W (torch dxd array): The Kleene star matrix
                    
        """
        d = self.dim
        
        W=self.linears[0].bias.data.view(d,-1)
        
        for i in range(1,len(self.linears)):
          W=torch.cat((W,self.linears[i].bias.data.view(d,-1)),1)
            
        W = torch.transpose(W,0,1).cpu().detach().numpy()  # [i, j]
        W=np.exp(W)
        
        return W

        
    
    
def custom_loss(output, target,n):  
     """
    Computes the loss function as in line 4 of Algorithm 1 in "Learning Bayesian Networks 
    from Extreme Data"
    
    Returns:
         loss (torch float): The loss value
                
    """

    loss = torch.sum(torch.abs(output-target))/n

    return loss

def get_extreme_data(X: np.ndarray, alpha: float = 0.85):
    
    """
    Computes the extreme data of X
    
    Args:
        X (n x d numpy array): Matrix of n d-dimensional observations
        alpha (float): quantile level in [0,1]
    
    Returns:
         list [X1,X2] (X1: n1 x d numpy array, X2: n1 x d numpy array): Set of explanatory
         and response variables as described in "Learning Bayesian Networks from Extreme Data",
         equation (12) and the subsequent text              
    """
    
    X2=np.copy(X)
    n=X.shape[0]
    d=X.shape[1]
   
    quantiles=np.nanquantile(X,alpha, axis=0)


    for j in range(d):
        X[(X[:,j]<quantiles[j]),j]=-np.inf

    
    ind=[]
    
    X=np.nan_to_num(X,nan=-np.inf, neginf=-np.inf)
    X2=np.nan_to_num(X2,nan=100)

    for i in range(n):
        if np.max(X[i,:])>-np.inf:
            ind+=[i]
    
    X=X[ind,:]
    X2=X2[ind,:]
    
    return[X,X2]




def trim(W_est: np.ndarray, w_threshold: float = 0.5):
    """
    Sets some values of W_est smaller than a threshold equal to zero
    
    Args:
        W_est (d x d numpy array): Computed Kleene star matrix
        w_threshold (float): Cut value
        
    
    Returns:
         B_est (d x d numpy array): All values in W_est smaller than w_threshold are set to zero
         If the resulting matrix is not supported on a DAG, cut the smallest values to obtain a DAG
         (see function trim2) and return its Kleene star matrix
    """


    W_est[W_est<w_threshold]=0

    
    G=nx.DiGraph(W_est)
    
    if nx.is_directed_acyclic_graph(G) == False:
        
        W_est=trim2(W_est)
        W_est[W_est>10**-8]=1
        B_est=fromCtoB(W_est)        
        return(B_est)
    
    else: 
        W_est[W_est>=w_threshold]=1        
        B_est=fromCtoB(W_est)        
        return(B_est)


def trim2(W_est: np.ndarray):
    """
    Set the smallest values of W_est necessary equal to zero to obtain a DAG
    
    Args:
        W_est (d x d numpy array): Computed Kleene star matrix        
    
    Returns:
         B_est (d x d numpy array): Sort values in W_est by size; Start with the largest values in 
         W_est and iteratively add them to a matrix of zeros. If one value causes the matrix to 
         contain a cycle, do not include it. Return the resulting matrix
    """


    val=np.argsort(W_est,axis=None)[::-1]
    d=W_est.shape[0]
    
    B_est=np.zeros((d,d))
    
    for ind in val:
     
        B_est[ind//d,ind%d]=W_est[ind//d,ind%d]
        
        G=nx.DiGraph(B_est)
        
        if nx.is_directed_acyclic_graph(G) == False:
            B_est[ind//d,ind%d]=0
        
        
    return(B_est)



def maxlinear_fit(model: nn.Module,
                      X: np.ndarray,
                      X2: np.ndarray,
                      E: bool = True,
                      lambda1: float = 0.3,
                      max_iter: int = 10000,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16):
    
    """
    Main maxlinear fit function
    
    Args:
        model (torch.nn.Module): A model 
        X1 (n x d numpy array): matrix of explanatory variables as in the function get_extreme_data
        X2 (n x d numpy array): matrix of response variables as in the function get_extreme_data
        E (bool): If True, calculate C^(d-1)*X, otherwise the computationally more feasbile C*X
        lambda1 (float): value for lambda as in line 4 of algorithm 1 in "Learning Bayesian Networks from Extreme Data"
        max_iter (ind): Maximum number of iterations in the gradient descent
        h_tol(float): The optimization terminates if the h value is smaller than h_tol 
                      (Recall: the h value is zero if and only if the result is supported on a DAG)
        rho(float): The maximum value of rho to enforce the h_val to be zero
    
    Returns:
         W_est (d x d numpy array): (d x d numpy array): Estimated Kleene star matrix
    """

    
    n=X.shape[0]

        
    l_r=lambda1*50
    
    

    
    
    
    for i, l in enumerate(model.linears):
    
        l.weight.data.fill_(1)
        l.bias.data.fill_(-1)
        l.bias.data[i]=-torch.inf
        l.weight.requires_grad=False
    
    
    
    
    #Initialization
    rho=10**-3
    
    
    
    optimizer = optim.SGD(model.parameters(), lr=l_r, momentum=0.5)
    
    rho_inc=0
    
    def closure():
        
             optimizer.zero_grad()
             
             #Restrict values in [0,1] in multiplicative case
             for i, layer in enumerate(model.linears):                            
                 model.linears[i].bias.data=torch.clamp(model.linears[i].bias.data,max=np.log(1))
    
             
             x_pred = model([torch.Tensor(X),torch.Tensor(X2)],E)
             
        
             loss  = custom_loss(x_pred, torch.Tensor(X2),n)+lambda1*model.weight_constraint()
             #print(loss)
             
             h_val = model.h_func()
             penalty = 0.5 * rho * h_val * h_val + rho*h_val
             
    
             
             primal_obj = loss + penalty

             primal_obj.backward()
    
            
             for i, layer in enumerate(model.linears):             
                 layer.bias.grad.data[i]=0
    
             return primal_obj
    
    for t in range(max_iter):
        
         h_val=model.h_func()
    
         if (h_val < h_tol or rho >rho_max) and t>300:
             #print("Terminated by h_val")
             for i, layer in enumerate(model.linears):
                                
                 model.linears[i].bias.data=torch.clamp(model.linears[i].bias.data,max=np.log(1))
                 
             break
      
    
         optimizer.step(closure)
           
        
         if model.h_func()>0.5*h_val:
             rho_inc+=1
             if rho_inc>=100:
                 rho_inc=0
                 rho*=2
            
    
    W_est=model.layer_to_adj()
    

    
    
    
    return W_est



if __name__ == '__main__':
    
    """
    The main function for the estimator
    for max-linear Bayesian networks (MLDAG) without automated parameter selection. 
    
    """
    
    #possible datasets: danube, top-colorado, middle-colorado, bottom-colorado, bottom-colorado150    
    
    dataset="danube"
    
    data_file = os.path.join(dataset,'data.pk')
    
    with open(data_file, 'rb') as file_name:
          df,labels,G_true = pickle.load(file_name)   
    
    X=np.array(df)
    d=X.shape[1]
    

    

    [X,X2]=get_extreme_data(X,alpha=0.8)
    
    
    
    
    model = Network(dim=d)
    
    
    W_est=trim2(maxlinear_fit(model,X, X2, E=True, lambda1= 0.5))
    
    with np.errstate(divide='ignore'):
         w_flat=np.log(W_est.flatten())
    w_flat=w_flat[np.isfinite(w_flat)]

    #B_est=trim(np.copy(W_est),w_threshold=np.quantile(W_est,0.8))

    B_est=trim(np.copy(W_est),w_threshold=np.exp(np.mean(scipy.cluster.vq.kmeans(w_flat,2)[0])))
    
    B=fromCtoB(G_true)
    
    
    
    save_folder="results"
    fname=dataset+".pkl"
    

    scores = count_accuracy(B,[B_est])
    print("Results for "+ dataset)
    print("True Positive Rate: ", scores[0]['tpr'])
    print("nSHD: ",scores[0]['shd'])
    print("False Discovery Rate: ", scores[0]['fdr'])
    print("False Positive Rate: ",scores[0]['fpr'])
    
    
    saveTo([X,B_est,B,scores[0]],save_folder,fname)
    
    

   