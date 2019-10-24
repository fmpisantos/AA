from sklearn.neighbors.kde import KernelDensity;
from sklearn.naive_bayes import GaussianNB;
from sklearn.svm import SVC;
import numpy as np;
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold;

def calcKDE(train,bandwith,feat):
	print(train[:,[feat]]);
    #return KernelDensity(kernel='gaussian', bandwidth=bandwith).fit(train[:,[feat]]);
    
def calcB(Xs,Ys, train, validation, bandwith):
    trainFalse = [];
    trainTrue = [];
    for i in train:
        if Ys[i] == 1.000:
            trainTrue.append(list(Xs[i]));
        else:
            trainFalse.append(list(Xs[i]));
    lnTrue = np.log(len(trainTrue)/len(train));
    lnFalse = np.log(len(trainFalse)/len(train));
    print(calcKDE(trainTrue,bandwith,1));
    '''
    for f in range(4):
    	for c in range(2):
    		if(c==0):
    			print(calcKDE(trainTrue,bandwith,f))
    		else:
    			print(calcKDE(trainFalse,bandwith,f))
    
    toSum = calcKDE(trainTrue,validation,bandwith);
	summ = sum(toSum);
    falseSum = summ + lnFalse;
    trueSum = summ + lnTrue;
    if falseSum > trueSum:
        return falseSum;
    else:
        return trueSum;
    '''
    
def kFolds(Ys,Xs,k,values):
    kf = StratifiedKFold(k);
    for train,valid in kf.split(Ys,Ys):
        calcB(Xs,Ys,train, valid,0.02);


def stats(values):
    Ys = np.array(values)[:,4];
    Xs = np.array(values)[:,:4];
    means = np.mean(Xs,0);
    stdevs = np.std(Xs,0);
    Xs = (Xs-means)/stdevs;
    return Xs,Ys;

def getValuesFromFile(fileName):
    text = open(fileName).readlines();
    values = [];
    for lin in text:
        va = lin.split('\t');
        va[4] = va[4].split('\n')[0];
        for i, l in enumerate(va):
            va[i] = float(l);
        values.append(va);
    shuffle(values);
    Xs,Ys = stats(values);
    kFolds(Ys,Xs,5,values);
    return values;

getValuesFromFile("TP1_train.tsv");
