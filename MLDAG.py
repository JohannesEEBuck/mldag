#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:35:30 2020

@author: JohannesBuck

The estimator
for max-linear Bayesian networks (MLDAG)
without automated parameter selection. 

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


class Network(nn.Module):
    def __init__(self, dim):
        super(Network, self).__init__()
        d=dim
        self.linears = nn.ModuleList([nn.Linear(1, d) for i in range(d)])
        self.dim = dim
    
    def forward(self, x, C_norm): 
        
        d = self.dim
        n= x[0].size()[0]
        X2=x[1]
        
        y=torch.zeros((d,d,n))

    
        for i, l in enumerate(self.linears):

            y[i,:,:] = torch.transpose(l(x[0][:,i].view(-1,1)),0,1)+torch.transpose(C_norm[i,:].repeat(n,1),0,1)

                
        x[0]=torch.cat((torch.transpose(x[1],0,1).view(-1,d,n), y), 0)
        output=torch.transpose(torch.max(x[0], axis=0).values,0,1)
        
        return output

    
    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dim
        
        W=torch.exp(self.linears[0].bias).view(d,-1)
        
        for i in range(1,len(self.linears)):
          W=torch.cat((W,torch.exp(self.linears[i].bias.view(d,-1))),1)
            
        W.fill_diagonal_(0)
            
        E = torch.matrix_exp(torch.mul(W,W))  # (Zheng et al. 2018)
        h = torch.trace(E) - d
        
        return h
    
    

    
    def weight_constraint(self):  
        reg=0
        
        for i, l in enumerate(self.linears):
                reg+=torch.sum(torch.exp(l.bias))
                 
        return (-torch.log(reg))

    
    @torch.no_grad()
    def layer_to_adj(self) -> np.ndarray:
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dim
        
        W=self.linears[0].bias.data.view(d,-1)
        
        for i in range(1,len(self.linears)):
          W=torch.cat((W,self.linears[i].bias.data.view(d,-1)),1)
            
        W = torch.transpose(W,0,1).cpu().detach().numpy()  # [i, j]
        W=np.exp(W)
        
        return W

        
    
    
def custom_loss(output, target,n):  

    loss = torch.sum(torch.abs(output-target))/n

    return loss

def get_extreme_data(X: np.ndarray, alpha: float = 0.85):
    
    X2=np.copy(X)
    n=X.shape[0]
    d=X.shape[1]
   
    quantiles=np.nanquantile(X,alpha, axis=0)


    for j in range(d):
        X[(X[:,j]<quantiles[j]),j]=-np.inf
    

    
    C_norm=np.zeros((d,d))      
        
    for i in range(d):
        for j in range(d):
            C_norm[i,j]=np.nanmean(X2[(X[:,i]>quantiles[i]),j])
    
    C_norm=torch.Tensor(C_norm) 

    for j in range(d):
        X[:,j]-=np.ma.masked_invalid(X[:,j]).mean()

    ind=[]
    
    X=np.nan_to_num(X,nan=-np.inf, neginf=-np.inf)
    X2=np.nan_to_num(X2,nan=100)

    for i in range(n):
        if np.max(X[i,:])>-np.inf:
            ind+=[i]
    
    X=X[ind,:]
    X2=X2[ind,:]
    
    return[X,X2,C_norm]




def trim(W_est: np.ndarray,
        w_threshold: float = 0.5):


    W_est[W_est<w_threshold]=0

    
    G=nx.DiGraph(W_est)
    
    if nx.is_directed_acyclic_graph(G) == False:
        
        W_est=idempotent(W_est)
        W_est[W_est>10**-8]=1
        B_est=fromCtoB(W_est)        
        return(B_est)
    
    else: 
        W_est[W_est>=w_threshold]=1        
        B_est=fromCtoB(W_est)        
        return(B_est)


def idempotent(W_est: np.ndarray):

    val=np.unique(np.sort(W_est, axis=None))
    
    B_est=np.copy(W_est)
    
    for v in val:
     
        B_est[B_est<=v]=0
        
        G=nx.DiGraph(B_est)
        
        if nx.is_directed_acyclic_graph(G) == False:
            continue
        
        else:            
            break
        
    return(B_est)



def maxlinear_fit(model: nn.Module,
                      X: np.ndarray,
                      X2: np.ndarray,
                      C_norm: np.ndarray,
                      lambda1: float = 0.3,
                      max_iter: int = 10000,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16):
    
            
   
    #[X,X2,C_norm]=get_extreme_data(X,alpha)

    
    n=X.shape[0]

        
    l_r=lambda1*50
    
    

    
    
    
    for i, l in enumerate(model.linears):
    
        l.weight.data.fill_(1)
        l.bias.data.fill_(-1)
        l.bias.data[i]=-torch.inf
        l.weight.requires_grad=False
    
    
    
    
    #Initialization
    rho=10**-5
    
    
    
    optimizer = optim.SGD(model.parameters(), lr=l_r, momentum=0.5)
    
    rho_inc=0
    
    def closure():
        
             optimizer.zero_grad()
             
             #Restrict values in [0,1] in multiplicative case
             for i, layer in enumerate(model.linears):                            
                 model.linears[i].bias.data=torch.clamp(model.linears[i].bias.data,max=np.log(1))
    
             
             x_pred = model([torch.Tensor(X),torch.Tensor(X2)], C_norm)
             
        
             loss  = custom_loss(x_pred, torch.Tensor(X2),n)+lambda1*model.weight_constraint()
    
             h_val = model.h_func()
             penalty = 0.5 * rho * h_val * h_val + rho*h_val
             
    
             
             primal_obj = loss + penalty

             primal_obj.backward()
    
            
             for i, layer in enumerate(model.linears):             
                 layer.bias.grad.data[i]=0
    
             return primal_obj
    
    for t in range(max_iter):
        
         h_val=model.h_func()
    
         if h_val < h_tol or rho >rho_max:
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
    
    #possible datasets: danube, top-colorado, middle-colorado, bottom-colorado, bottom-colorado150    
    
    dataset="top-colorado"
    
    data_file = os.path.join(dataset,'data.pk')
    
    with open(data_file, 'rb') as file_name:
          df,labels,G_true = pickle.load(file_name)   
    
    X=np.array(df)

    

    [X,X2,C_norm]=get_extreme_data(X,alpha=0.75)
    
    
    d=X.shape[1]
    
    model = Network(dim=d)
    
    
    W_est=maxlinear_fit(model,X, X2, C_norm, lambda1= 0.3)
    
    B_est=trim(np.copy(W_est),w_threshold=np.quantile(W_est,0.8))
    
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
    
    

   
   
        
        









        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        