# -*- coding: utf-8 -*-
""" The Simulation Study for MLDAG with a max-linear model and a misspecified, 
additive model to check for robustness

Parameters
----------
reps: int
    number of repetitions
drange: list 
    list of number of nodes to consider in the simulation study, e.g. [10,20,30]
nrange list 
    list of number of observations to consider in the simulation study, e.g. [50,100,250,500]
rrange list 
    list of noise-to-signal ratios to consider in the simulation study, e.g. [0.1,0.3,0.5
p float
    probability to draw an edge between a pair of nodes in the Erdos Renyi Graph - must be between 0 and 1                                                                             ]
  
Returns
-------
scores_all_params: dict
  dictionary of scores; Scores are also saved as scores_all_params.pk in the simulation subfolder

"""
if __name__ == "__main__":

    from utils import *
    from MLDAG import *
    import scipy.stats
    
    random.seed(1)
    np.random.seed(1)
    
    reps=100
    drange=[10,20,30]
    nrange=[50,100,250,500]
    rrange=[0.1,0.3,0.5]
    p=0.3
    
    scores_coll={}
    
    for sample in range(reps):
        print(sample)
        for d in drange:
            
            C=np.multiply(np.random.binomial(1, p, size=(d,d)),np.random.uniform(low=0.1, high=1.0, size=(d,d)))               
            C[np.tril_indices(d)]=0               
            B=fromCtoB(C)
            B2=np.copy(B)
            B2[B2>0]=1
            
            s=np.random.uniform(low=0.1, high=1.0, size=(1,d))
            
            for n in nrange:
                scale=np.repeat(s,n, axis=0)
                Z=np.multiply(scipy.stats.invweibull.rvs(1,size=(n,d)),scale)
                X=np.log(MLMatrixMult(Z,B))
                X_add=np.zeros((n,d))
                    
                for i in range(n):
                    for j in range(d):
                        X_add[i,j]=np.log(np.sum(np.multiply(Z[i,:],B[:,j]))) 
                        
                for r in rrange:
                    
                    noise=np.random.normal(0,r*np.mean(np.std(X,axis=0)),size=(n,d))
                    X+=noise
                    
                    noise2=np.random.normal(0,r*np.mean(np.std(X_add,axis=0)),size=(n,d))
                    X_add+=noise
                    
                    [X1,X2]=get_extreme_data(np.copy(X),alpha=0.8)
                    
                    [X3,X4]=get_extreme_data(np.copy(X_add),alpha=0.8)
                    
                        
                        
                    d=X.shape[1]
                    
                    model = Network(dim=d)
                    
                    
                    W_est=trim2(maxlinear_fit(model,X1, X2,E=False, lambda1= 0.3))
                    with np.errstate(divide='ignore'):
                        w_flat=np.log(W_est.flatten())
                    w_flat=w_flat[np.isfinite(w_flat)]
                                 
                    B_est=trim(np.copy(W_est),w_threshold=np.exp(np.mean(scipy.cluster.vq.kmeans(w_flat,2)[0])))
                    
                    scores = count_accuracy(B2,[B_est])        
                    scores_coll[(sample, d, n, r,0)]=scores[0]
    
                                    
                    d=X3.shape[1]
                    
                    model = Network(dim=d)
                    
                    
                    W_est=trim2(maxlinear_fit(model,X3, X4,E=False, lambda1= 0.3))
                    with np.errstate(divide='ignore'):
                        w_flat=np.log(W_est.flatten())
                    w_flat=w_flat[np.isfinite(w_flat)]
                    
                    
                    B_est=trim(np.copy(W_est),w_threshold=np.exp(np.mean(scipy.cluster.vq.kmeans(w_flat,2)[0])))
                    
                    scores = count_accuracy(B2,[B_est])
                    scores_coll[(sample, d, n, r,1)]=scores[0]
    
                    
                    
    saveTo(scores_coll,"simulation","scores_all_params.pk")

