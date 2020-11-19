#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file implements the HRP algorithm, the MinVAr portfolio and the IVP portf.

Code for HRP is based on Lopez de Prado, M. (2018). Advances in Financial 
Machine Learning. Wiley. The code has been adapted in order to be used with
python 3 and the data set.
 
@author: Souhir Ben Amor
@date: 20201010
"""

#[0Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#[0]Upload input data, Financial Institutions of the 6 Emerging Markets
FIs_prices = pd.read_excel("Financial Institutions Price Series.xlsx")
print(FIs_prices)

# In[1]: 
# Load modules

import os
path = os.getcwd() # Set Working directory here

# Import modules for Datastructuring and calc.
import pandas as pd
import numpy as np
from scipy import stats
import warnings
from tqdm import tqdm

# Modules for RHP algorithm
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch

# Modules for the network plot
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix

# Modules for Markowitz optimization
import cvxopt as opt
import cvxopt.solvers as optsolvers

warnings.filterwarnings("ignore") # suppress warnings in clustering


print(path)

# In[2]: 
# define functions for HRP and IVP

def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp


def getClusterVar(cov, cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems, cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar


def getQuasiDiag(link):
    # Sort clustered items by distance
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] # number of original items
    while sortIx.max() >=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
        df0=sortIx[sortIx>=numItems] # find clusters
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1], index=i+1)
        sortIx=sortIx.append(df0) # item 2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()


def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)//2),(len(i)//2,\
                len(i))) if len(i)>1] # bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1 
    # This is a proper distance metric
    dist=((1-corr)/2.)**.5 # distance matrix
    return dist


def plotCorrMatrix(path, corr, labels=None):
    # Heatmap of the correlation matrix
    if labels is None:labels=[]
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.xticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.savefig(path,dpi=300, transparent=True)
    mpl.clf();mpl.close() # reset pylab
    return


# In[3]: 
# define function for MinVar portfolio
    
# The MIT License (MIT)
#
# Copyright (c) 2015 Christian Zielinski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.   

def min_var_portfolio(cov_mat, allow_short=False):
    """
    Computes the minimum variance portfolio.

    Note: As the variance is not invariant with respect
    to leverage, it is not possible to construct non-trivial
    market neutral minimum variance portfolios. This is because
    the variance approaches zero with decreasing leverage,
    i.e. the market neutral portfolio with minimum variance
    is not invested at all.
    
    Parameters
    ----------
    cov_mat: pandas.DataFrame
        Covariance matrix of asset returns.
    allow_short: bool, optional
        If 'False' construct a long-only portfolio.
        If 'True' allow shorting, i.e. negative weights.

    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """
    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError("Covariance matrix is not a DataFrame")

    n = len(cov_mat)
        
    P = opt.matrix(cov_mat.values)
    q = opt.matrix(0.0, (n, 1))

# Constraints Gx <= h
    if not allow_short:
    # x >= 0
        G = opt.matrix(-np.identity(n))
        h = opt.matrix(0.0, (n, 1))
    else:
        G = None
        h = None

# Constraints Ax = b
# sum(x) = 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Solve
    optsolvers.options['show_progress'] = False
    sol = optsolvers.qp(P, q, G, h, A, b)
        
    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")
            
 # Put weights into a labeled series
    weights = pd.Series(sol['x'], index=cov_mat.index)
    return weights

# In[4]: 
# Define functions for network graphs
    
#Function to plot Network plots
def plotNetwork(path,corr):
    # Transform it in a links data frame
    #links=corr.stack().reset_index()
    #Build graph
    corr=Corr_mat
    adj_matrix = corr
    constits_latest = corr.index
    # remove self-loops
    adj_matrix = np.where((adj_matrix<=1.000001) & (adj_matrix>=0.99999),0,adj_matrix)
    # replace values that are below threshold
    # create undirected graph from adj_matrix
    graph = from_numpy_matrix(adj_matrix, parallel_edges=False, create_using= nx.Graph())
    # set names to crypots
    graph = nx.relabel.relabel_nodes(graph, dict(zip(range(len(constits_latest)), constits_latest)))
    pos_og =  nx.circular_layout(graph, scale=2)
    pos = nx.circular_layout(graph, scale=1.7)
    
    for p in pos:  # raise text positions
        if pos[p][1]>1:
            pos[p][1] += 0.15
        if pos[p][1]<-1:
            pos[p][1] -= 0.15
        elif pos[p][0]<0:
            pos[p][0] -= 0.3
        else:
            pos[p][0]+=0.3
    plt = mpl.figure(figsize = (5,5)) 
    nx.draw(graph, pos_og, with_labels= False)
    nx.draw_networkx_labels(graph, pos)
     
    plt.savefig(path,dpi=300 ,transparent=True)
    mpl.clf();mpl.close()
    return


## In[5]:
# Loading and structuring FIs_prices data sets

FIs_prices = FIs_prices[(~FIs_prices.isnull()).all(axis=1)] # Deleting empty rows

##FIs_prices["date"] =FIs_prices['date'].map(lambda x: str(x)[:-9]) # Removing timestamp
FIs_prices = FIs_prices.rename(columns = {"date":"Date"})
FIs_prices = FIs_prices.replace(to_replace = 0, method = "ffill")
Price_data_univ=FIs_prices
Price_data_univ = Price_data_univ.set_index("Date") # define Date  as index
# Calculating returns 
Return_data_univ = Price_data_univ.pct_change() #calculate daily returns
Return_data_univ = Return_data_univ.drop(Return_data_univ.index[range(0,1)])

# Calculating covariance matrix

Cov_mat = Return_data_univ.cov() # Covariance matrix of the return matrix
Corr_mat=Return_data_univ.corr() # Correlation matrix of the return matrix

print(Corr_mat)
print(Cov_mat)
print(Return_data_univ)


# In[6]:
# Heatmap and network analysis of corr. matrix

# Plotting Correlation matrix heatmap

plotCorrMatrix(path+"/Corr_Heatmap_FIs_unsorted",Corr_mat)


# network plot of correlation matrix

plotNetwork(path+"/Corr_Network_FIs_unsorted.png", Corr_mat)

# Sort correlation matrix
dist=correlDist(Corr_mat)
link=sch.linkage(dist,'single')
sortIx=getQuasiDiag(link) 
sortIx=Corr_mat.index[sortIx].tolist() # recover labels 
Corr_sorted=Corr_mat.loc[sortIx,sortIx] # reorder

# Plot sorted correlation matrix
plotCorrMatrix(path+"/Corr_Heatmap_FIs_sorted",Corr_sorted)

# Plot dendogram of the constituents
#2) Cluster Data
mpl.figure(num=None, figsize=(20, 10), dpi=300, facecolor='w', edgecolor='k')    
dn = sch.dendrogram(link, labels = dist.columns)
mpl.savefig(path+"/Dendrogram_FIs.png", transparent = True, dpi = 300)
mpl.clf();mpl.close() # reset pylab

print(plotNetwork)

# In[7]:
#Function to calculate the HRP portfolio weights

def HRPportf(cov,corr):
    #1) Cluster covariance matrix
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link) 
    sortIx=corr.index[sortIx].tolist() # recover labels
    #2) Allocate capital according to HRP
    weights_hrp=getRecBipart(cov,sortIx)
    return weights_hrp

# In[8]:
# Compute the weights for the Markowitz MinVar and the HRP portfolio and the 
# IVP portfolio

w_HRP=np.array([HRPportf(Cov_mat,Corr_mat).index,HRPportf(Cov_mat,Corr_mat).round(3)])
w_HRP=pd.DataFrame(np.transpose(w_HRP))
w_HRP.columns = ["Asset","Weights HRP"]

w_MinVar= np.array([min_var_portfolio(Cov_mat).index,min_var_portfolio(Cov_mat).round(3)])
w_MinVar=pd.DataFrame(np.transpose(w_MinVar))
w_MinVar.columns = ["Asset","Weights MinVar"]

w_IVP= np.array([Cov_mat.index, getIVP(Cov_mat).round(3)])
w_IVP=pd.DataFrame(np.transpose(w_IVP))
w_IVP.columns = ["Asset","Weights IVP"]

Weights = pd.merge(w_MinVar,w_IVP,\
                   on="Asset", how = "inner")
Weights = pd.merge(Weights,w_HRP,\
                   on="Asset", how = "inner")

print(Weights.to_latex(index=True)) # Latex table output

# In[9]:
# Backtesting the three optimisation methods for FIs_prices dataset

# Function to calculate the weigths in sample and then test out of sample
def Backtest_FIs_prices(returns, rebal = 30): # rebal = 30 default rebalancing after 1 month
    nrows = len(returns.index)-rebal # Number of iterations without first set to train
    rets_train = returns[:rebal]
    
    cov,corr = rets_train.cov(), rets_train.corr()
    w_HRP=np.array([HRPportf(cov,corr).index,HRPportf(cov,corr)])
    w_HRP=pd.DataFrame(np.transpose(w_HRP))
    w_HRP.columns = ["Asset","Weights HRP"]

    w_MinVar= np.array([cov.index,min_var_portfolio(cov)])
    w_MinVar=pd.DataFrame(np.transpose(w_MinVar))
    w_MinVar.columns = ["Asset","Weights MinVar"]

    w_IVP= np.array([cov.index, getIVP(cov)])
    w_IVP=pd.DataFrame(np.transpose(w_IVP))
    w_IVP.columns = ["Asset","Weights IVP"]
                     
    Weights = pd.merge(w_MinVar, w_IVP, on="Asset", how = "inner")
    Weights = pd.merge(Weights,w_HRP, on="Asset", how = "inner")
    Weights = Weights.drop(Weights.columns[0],axis=1).to_numpy() 
    

    portf_return = pd.DataFrame(columns=["MinVar","IVP","HRP"], index = range(nrows))
    
    for i in tqdm(range(rebal,nrows+rebal)):
    
            if i>rebal and i<nrows-rebal and i % rebal == 0: # Check for rebalancing date
                rets_train = returns[i-rebal:i]
                cov,corr = rets_train.cov(), rets_train.corr()
                w_HRP=np.array([HRPportf(cov,corr).index,HRPportf(cov,corr)])
                w_HRP=pd.DataFrame(np.transpose(w_HRP))
                w_HRP.columns = ["Asset","Weights HRP"]
                
                w_MinVar= np.array([cov.index,min_var_portfolio(cov)])
                w_MinVar=pd.DataFrame(np.transpose(w_MinVar))
                w_MinVar.columns = ["Asset","Weights MinVar"]
            
                w_IVP= np.array([cov.index, getIVP(cov)])
                w_IVP=pd.DataFrame(np.transpose(w_IVP))
                w_IVP.columns = ["Asset","Weights IVP"]
                     
                Weights = pd.merge(w_MinVar, w_IVP, on="Asset", how = "inner")
                Weights = pd.merge(Weights,w_HRP, on="Asset", how = "inner")
                Weights = Weights.drop(Weights.columns[0],axis=1).to_numpy()     
       
            

            portf_return["MinVar"][i-int(rebal):i-int(rebal)+1]=np.average(returns[i:i+1].to_numpy(),\
                    weights = Weights[:,0].reshape(len(returns.columns),1).ravel(), axis = 1)
            portf_return["IVP"][i-int(rebal):i-int(rebal)+1]=np.average(returns[i:i+1].to_numpy(),\
                    weights = Weights[:,1].reshape(len(returns.columns),1).ravel(), axis = 1)
            portf_return["HRP"][i-int(rebal):i-int(rebal)+1]=np.average(returns[i:i+1].to_numpy(),\
                    weights = Weights[:,2].reshape(len(returns.columns),1).ravel(), axis = 1)
           
            
    return portf_return

# In[10]:    
# Calculate the backtested portfolio returns
    
portf_rets = Backtest_FIs_prices(Return_data_univ, rebal=30)
portf_rets2 = 1+portf_rets

rebal = 30
# Calculate index
portf_index = pd.DataFrame(columns=["MinVar","IVP","HRP"], \
                           index = Return_data_univ.index)[rebal:]
portf_index[0:1] = 100
for i in range(1,len(portf_index.index)):
    portf_index["MinVar"][i:i+1] = float(portf_index["MinVar"][i-1:i]) * float(portf_rets2["MinVar"][i-1:i])
    portf_index["IVP"][i:i+1] = float(portf_index["IVP"][i-1:i]) * float(portf_rets2["IVP"][i-1:i])
    portf_index["HRP"][i:i+1] = float(portf_index["HRP"][i-1:i]) * float(portf_rets2["HRP"][i-1:i])

index = portf_index.plot.line()
mpl.savefig(path+"/Index_FIs_prices.png", transparent = True, dpi = 300)
mpl.clf();mpl.close() # reset pylab


# Calculate portfolio return and portfolio variance
mean_MinVar = stats.mstats.gmean(np.array(portf_rets2["MinVar"], dtype=float))-1
mean_IVP = stats.mstats.gmean(np.array(portf_rets2["IVP"], dtype=float))-1
mean_HRP = stats.mstats.gmean(np.array(portf_rets2["HRP"], dtype=float))-1

# Daily Standard deviation
std_MinVar = portf_rets["MinVar"].std()
std_IVP = portf_rets["IVP"].std()
std_HRP = portf_rets["HRP"].std()

# Sharpe ratios
SR_MinVar = mean_MinVar/std_MinVar
SR_IVP = mean_IVP/std_IVP
SR_HRP = mean_HRP/std_HRP

Perf_figures = pd.DataFrame([[mean_MinVar, mean_IVP,mean_HRP], [std_MinVar, \
                            std_IVP, std_HRP],[SR_MinVar,SR_IVP,SR_HRP]], \
    index =['Mean', 'Std', 'SR'], columns = ["MinVar","IVP","HRP"])

print(Perf_figures.to_latex(index=True)) # Latex table output

