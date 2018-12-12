#you can use this program as well https://medium.com/100-days-of-algorithms/day-97-locally-weighted-regression-c9cfaff087fb
#video https://www.youtube.com/watch?v=ZSWs9Kj_10I

import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np1
import numpy.linalg as np
from scipy.stats.stats import pearsonr
import pdb
def kernel(point, xmat, k):
	m,n=np1.shape(xmat)  #size of matrix m
	weights=np1.mat(np1.eye(m)) #np.eye returns mat with 1 in the diagonal 
	for j in range(m):
		diff=point-xmat[j]
		weights[j,j]=np1.exp(diff*diff.T/(-2.0*k**2))
	return weights

def localWeight(point,xmat,ymat,k):
	wei=kernel(point,xmat,k)
	W=(xmat.T*(wei*xmat)).I*(xmat.T*(wei*ymat.T))
	return W

def localWeightRegression(xmat,ymat,k):
	row,col=np1.shape(xmat) #return 244 rows and  2 columns
	ypred=np1.zeros(row)
	for i in range(row):
		ypred[i]=xmat[i]*localWeight(xmat[i],xmat,ymat,k)
	return ypred

data=pd.read_csv('tips_10.csv')
bill=np1.array(data.total_bill)
tip=np1.array(data.tip)

mbill=np1.mat(bill)
mtip=np1.mat(tip)

mbillMatCol=np1.shape(mbill)[1] # 1 for vertical i.e columns
onesArray=np1.mat(np1.ones(mbillMatCol))
xmat=np1.hstack((onesArray.T,mbill.T)) #hstack concate horizontal lists it takes one value from the fist and one from the second  
print(xmat)

ypred=localWeightRegression(xmat,mtip,2)
SortIndex=xmat[ :,1].argsort(0) #argsort take the index of each and sort them according to the orginal value 
xsort=xmat[SortIndex][:,0]

fig= plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(bill,tip,color='blue')
ax.plot(xsort[:,1],ypred[SortIndex],color='red',linewidth=1)
plt.xlabel('Total bill')
plt.ylabel('tip')
plt.show();
pdb.set_trace()
