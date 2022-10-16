#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 06 12:35:30 2022

@author: JohannesBuck

The estimator
for max-linear Bayesian networks (MLDAG) with automated parameter selection. 
(cf Algorithm 2 in "Learning Bayesian Networks from Extreme Data" from Buck and Tran)

"""

import numpy as np
import networkx as nx
import pickle
import os
from MLDAG import trim, trim2, maxlinear_fit, Network, get_extreme_data
from utils import create_sa, fromCtoB, MLMatrixMult, count_accuracy, _count_accuracy, saveTo, plotParameterSelection
import random
import scipy.cluster



def generateSubsamples(X,save_folder, nrep):
    
"""
Function for creating  subsamples of X with replacement 

Args: 
   X (n x d numpy array): Matrix of n d-dimensional observations
   save_folder (str): name of the subfolder to save subsamples
   nrep (int): number of subsamples
 
 Returns: 
   samples (list of n x d numpy arrays): list with n_rep subsamples of X

"""

  n=X.shape[0]
  samples=[]
  
  samples=[np.array(random.choices(X, k = n)) for i in range(nrep)]
  
  saveTo(samples,save_folder,'samples.pk')    

  return samples


def mldagAutoSelect(X, save_folder, n_rep = 100, alpha_range = [0.7,0.8,0.9], lambda_range = [0.3], saveAll=True): 

"""
Function for calculating MLDAG for all alpha in alpha_range and all lambda in lambda_range

Args: 
   X (n x d numpy array): Matrix of n d-dimensional observations
   save_folder (str): name of the subfolder to save subsamples
   n_rep (int): number of subsamples
   alpha_range (list of floats in [0,1]): All values of alpha to consider
   lambda_range (list of floats in [0,1]): All values of lambda to consider
   saveAll (bool): True, if all estimated Kleene start matrices should be saved, otherwise false 

 
 Returns: 
   B_est_coll(dict of list of d x d numpy arrays): Output is a dictionary. For each key (lambda,alpha),  
   B_est_coll contains a list of estimated Kleene star matrices for X and the n_rep subsamples

"""
    
  
  d=X.shape[1]  
  
  samples=[X]+generateSubsamples(X,save_folder, n_rep)

  B_est_coll={}

  for idx, sample in enumerate(samples):  
      if (idx%10) == 0:
          print("Generating Estimates for subsample ", idx, " out of ", n_rep)

      if idx==0:
          bool_E=True
      else:
          bool_E=False
        
      for alpha in alpha_range:
          
          
          [X,X2]=get_extreme_data(np.copy(sample),alpha)
            
          for lambda1 in lambda_range:
                 
            model = Network(dim=d)       
            W_est=trim2(maxlinear_fit(model,np.copy(X),np.copy(X2),bool_E, lambda1))
            
            with np.errstate(divide='ignore'):
                w_flat=np.log(W_est.flatten())
            w_flat=w_flat[np.isfinite(w_flat)]


            B_est=trim(np.copy(W_est),w_threshold=np.exp(np.mean(scipy.cluster.vq.kmeans(w_flat,2)[0])))
            
            
            if idx==0:
                B_est_coll[(lambda1,alpha)]=[B_est]
            else:
                B_est_coll[(lambda1,alpha)]+=[B_est]

                
                
  if saveAll:
      saveTo(B_est_coll,save_folder,'B_est_all_params.pk')
  
  return B_est_coll

  

if __name__ == "__main__":
    
"""
The main function for the estimator
for max-linear Bayesian networks (MLDAG) with automated parameter selection. 

"""
    
    
    #possible datasets: "danube", "top-colorado", "middle-colorado", "bottom-colorado", "bottom-colorado150"
    
    datasets=["top-colorado", "middle-colorado", "bottom-colorado", "bottom-colorado150"]
  
          
    alpharange = [0.8]
    lambdarange = [0.3,0.35,0.4,0.45,0.5]
    
        
    for dataset in datasets:
        
        random.seed(1)
        print("Automated Parameter Selection for ", dataset)
            
        data_file = os.path.join(dataset,'data.pk')
    
        with open(data_file, 'rb') as file_name:
            df,labels,G_true = pickle.load(file_name)   
    
        X=np.array(df)
        B=fromCtoB(G_true)
        

        B_est_coll=mldagAutoSelect(X,save_folder=dataset,alpha_range=alpharange, lambda_range=lambdarange)
    
        #if precalculated, you can also just load it
        # with open(os.path.join(dataset, "B_est_all_params_save.pk"), 'rb') as file:      
        #     B_est_coll=pickle.load(file) 
    
    


        param_best={"alpha": np.inf, "lambda": np.inf}

        shd_best=np.inf
    
        scores_coll={}
    
        for lambda1 in lambdarange:
            for alpha in alpharange:
                
                scores = count_accuracy(B_est_coll[(lambda1,alpha)][0],B_est_coll[(lambda1,alpha)])
                scores_coll[(lambda1,alpha)]=scores
                                    

                if np.mean([b["shd"] for b in scores])<shd_best:
        
                   shd_best=np.mean([b["shd"] for b in scores])
                   param_best['alpha']=alpha
                   param_best['lambda']=lambda1
                   
        
    
        print("The optimal parameters according to the parameter selection are: alpha=", param_best['alpha'], "and lambda=", param_best["lambda"])
        
        

        B_est=B_est_coll[(param_best['lambda'], param_best['alpha'])][0]
        
        scores = count_accuracy(B,[B_est])
        
        print("For the chosen parameters, we get the following scores")
        print("True Positive Rate: ", scores[0]['tpr'])
        print("nSHD: ",scores[0]['shd'])
        
        saveTo(scores_coll,dataset,'scores_all_params.pk')
        
        if len(lambdarange)==1 or len(alpharange)==1:
            plotParameterSelection(dataset,B, B_est_coll, lambdarange,alpharange)
        
        
        
        
        
    
    
    
  