import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import networkx as nx

def cost(bp):
    return (1-norm(p0,sdp[j]).cdf(bp))*7*dq+norm(p0,sdp[j]).cdf(bp)*bp*bq
                                                         

sdp = np.array([0.157639831715, 0.151055732908, 0.142867060282, 0.140828684561, 0.148690581036, 0.196368129973, 
 0.330699185229, 0.381459254495, 0.420121331093, 0.421702869329, 0.354668663841, 0.298185561582, 0.296323629387,
 0.273832453956, 0.280609754721, 0.264661413093, 0.266587165001, 0.27994932059, 0.305775153413, 0.265929461315,
 0.257406989075, 0.237027717294, 0.205583941283, 0.142060124204])

#res = minimize(cost, p0, method = 'nelder-mead', options = {'xtol': 1e-8, 'disp': False})
#pm = res.x


demand = pd.read_csv('Demand_LB_pred.csv', header = None).as_matrix()
price = pd.read_csv('Price_LB_pred.csv', header = None).as_matrix()
solar = pd.read_csv('Solar_LB_pred.csv', header = None).as_matrix()
k3 = 0
t = 50

#for i in range(t):
#    for j in range(24):

G = nx.MultiDiGraph()
#G.add_node(0)
#G.add_node(2)
#G.add_edge(0, 2)

for i in range(t):
    for j in range(24):
        for k in range(25*1+1):
            G.add_node((i*24+j)*100+k)

##for i in range(t):
##    for j in range(24):
##        for k in range(25*1+1):
##            for k2 in range(max(k-5,0),min(k+5,25)):
##                res=minimize(cost,price[i,j],method ='nelder-mead',options={'xtol':1e-8,'disp':False})   
##                G.add_edge((i*24+j)*100+k,(i*24+j+1)*100+k2, weight=res.fun)


bwgt=0
for i in range(t):
    for j in range(24):
        if i * j != 49 * 23:
            p0=price[i,j]
            
            for k in range(25*1+1):
                dq=max((demand[i,j]-solar[i,j]-0.8*min(k,5),0))
                
                k2 = max(k-5,0)
                bq = max(demand[i,j]-solar[i,j]+0.8*(k2-k),0)
                res=minimize(cost,price[i,j],method ='nelder-mead',options={'xtol':1e-8,'disp':False})
                bwgt=res.fun
                if res.x<=p0-0.2:
                    bq=bq+1
                    dq=dq+1
                    res=minimize(cost,price[i,j],method ='nelder-mead',options={'xtol':1e-8,'disp':False})
                    bq=bq-1
                    dq=dq-1
                    bwgt=cost(res.x)
                k3=int(round((1-norm(p0,sdp[j]).cdf(res.x))*(max(0,k-5,k-1.25*(demand[i,j]-solar[i,j])))+norm(p0,sdp[j]).cdf(res.x)*k2))
                G.add_edge((i*24+j)*100+k,(i*24+j+1)*100+k3, weight=bwgt, bprice = res.x, bquantity = bq)

                k2 = min(k+5,25)
                bq = max(demand[i,j]-solar[i,j]+k2-k,0)
                res=minimize(cost,price[i,j],method ='nelder-mead',options={'xtol':1e-8,'disp':False})
                bwgt=res.fun
                if res.x<=p0-0.2:
                    bq=bq+1
                    dq=dq+1
                    res=minimize(cost,price[i,j],method ='nelder-mead',options={'xtol':1e-8,'disp':False})
                    bq=bq-1
                    dq=dq-1
                    bwgt=cost(res.x)
                k3=int(round((1-norm(p0,sdp[j]).cdf(res.x))*(max(0,k-5,k-1.25*(demand[i,j]-solar[i,j])))+norm(p0,sdp[j]).cdf(res.x)*k2))
                G.add_edge((i*24+j)*100+k,(i*24+j+1)*100+k3, weight=bwgt, bprice = res.x, bquantity = bq)

                for l in range(-24,25):
                    bpr=norm(p0,sdp[j]).ppf(0.5+0.02*l)
                    bwgt=cost(bpr)
                    k3=int(round((0.5-0.05*l)*(max(0,k-5,k-1.25*(demand[i,j]-solar[i,j])))+(0.5+0.05*l)*k2))
                    G.add_edge((i*24+j)*100+k,(i*24+j+1)*100+k3, weight=bwgt, bprice = bpr, bquantity = bq)
                    
                k2 = k
                bq = max(demand[i,j]-solar[i,j]+k2-k,0)
                res=minimize(cost,price[i,j],method ='nelder-mead',options={'xtol':1e-8,'disp':False})
                bwgt=res.fun
                if res.x<=p0-0.2:
                    bq=bq+1
                    dq=dq+1
                    res=minimize(cost,price[i,j],method ='nelder-mead',options={'xtol':1e-8,'disp':False})
                    bq=bq-1
                    dq=dq-1
                    bwgt=cost(res.x)
                k3=int(round((1-norm(p0,sdp[j]).cdf(res.x))*(max(0,k-5,k-1.25*(demand[i,j]-solar[i,j])))+norm(p0,sdp[j]).cdf(res.x)*k2))
                G.add_edge((i*24+j)*100+k,(i*24+j+1)*100+k3, weight=bwgt, bprice = res.x, bquantity = bq)

                for l in range(-24,25):
                    bpr=norm(p0,sdp[j]).ppf(0.5+0.02*l)
                    bwgt=cost(bpr)
                    k3=int(round((0.5-0.05*l)*(max(0,k-5,k-1.25*(demand[i,j]-solar[i,j])))+(0.5+0.05*l)*k2))
                    G.add_edge((i*24+j)*100+k,(i*24+j+1)*100+k3, weight=bwgt, bprice = bpr, bquantity = bq)
                   
 
    print i
    

G.add_node('end')

i=49
j=23
for k in range(25*1+1):
	k2 = max(k-5,0)
	bq = max(demand[i,j]-solar[i,j]+0.8*(k2-k),0)                                
	res=minimize(cost,price[i,j],method ='nelder-mead',options={'xtol':1e-8,'disp':False})
	G.add_edge((i*24+j)*100+k,'end', weight=res.fun, bprice = res.x, bquantity = bq)
    
subm = pd.DataFrame(index=range(1200),columns=range(2))
ulti2=nx.dijkstra_path(G,0,'end')

for i in range(len(ulti2)-1):
    x=0
    temp=10000000
    for j in range(G.number_of_edges(ulti2[i], ulti2[i+1])):
        if G[ulti2[i]][ulti2[i+1]][j]['weight']<temp:
            temp=G[ulti2[i]][ulti2[i+1]][j]['weight']
            x=j
        subm.loc[i,0]=G[ulti2[i]][ulti2[i+1]][x]['bprice']
        subm.loc[i,1]=G[ulti2[i]][ulti2[i+1]][x]['bquantity']

for i in range(1200):
    subm.loc[i,0]=np.asscalar(subm.loc[i,0])

for i in range(1200):
	subm.loc[i,0]=round(subm.loc[i,0]*100)/100
	subm.loc[i,1]=round(subm.loc[i,1]*100)/100	



subm.to_csv('23.csv',header=False,index=False)
