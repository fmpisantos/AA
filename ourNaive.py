from sklearn.neighbors.kde import KernelDensity;
from sklearn.naive_bayes import GaussianNB;
from sklearn.svm import SVC;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.utils import shuffle;
from sklearn.model_selection import StratifiedKFold;
from sklearn.metrics import accuracy_score;

testX = testY = [[],[]];
best = [];

def calcKDE(train,bandwith,feat):
    return KernelDensity(kernel='gaussian', bandwidth=bandwith).fit(np.array(train)[:,[feat]]);
    
def calcB(Xs,Ys, train, validation, bandwith,test):
    trainFalse = [];
    trainTrue = [];
    for i in train:
        if Ys[i] == 1:
            trainTrue.append(list(Xs[i]));
        else:
            trainFalse.append(list(Xs[i]));
    lnTrue = np.log(len(trainTrue)/len(train));
    lnFalse = np.log(len(trainFalse)/len(train));
    kde = [[],[]]
    for f in range(4):
    	for c in range(2):
    		if(c==1):
    			kde[c].append(calcKDE(trainTrue,bandwith,f));
    		else:
    			kde[c].append(calcKDE(trainFalse,bandwith,f));
    if test:
        return (1-score(np.array(testX),np.array(testY),lnTrue,lnFalse,kde));
    else:
        return (1-score(Xs[train],Ys[train],lnTrue,lnFalse,kde)),(1-score(Xs[validation],Ys[validation],lnTrue,lnFalse,kde));
    
def predict(X, pTrue, pFalse,kde):
    probFalse = np.repeat(pFalse, X.shape[0]);
    probTrue = np.repeat(pTrue, X.shape[0]);
    global best;
    best = [];
    for i in range(X.shape[1]):
        probFalse += kde[0][i].score_samples(np.array(X)[:,[i]]);
        probTrue += kde[1][i].score_samples(np.array(X)[:,[i]]);
    for j in range(X.shape[0]):
        if probFalse[j]>probTrue[j]:
            best.append(0);
        else:
            best.append(1);
    return np.array(best);
    
def score(X,Y, pTrue, pFalse,kde):
    return accuracy_score(Y, predict(X,pTrue, pFalse,kde));
        
#Returns Best bandwidth
def kFolds(Ys,Xs,k,values):
    bandwidths = [];
    trainErrorA = [];
    trainValidA = [];
    kf = StratifiedKFold(k);
    bandwith = 0.02;
    bestBandwidth = 0;
    bestVError = 999999;
    while bandwith <= 0.6:
        tError = vError = 0;
        for train,valid in kf.split(Ys,Ys):
            trainError,validError = calcB(Xs,Ys,train, valid,bandwith,False);
            tError+=trainError;
            vError+=validError;
        if vError/k < bestVError:
            bestVError = vError/k;
            bestBandwidth = bandwith;
        bandwith+=0.02;
        bandwidths.append(bandwith);
        trainErrorA.append(tError/k);
        trainValidA.append(vError/k);
    '''
    plt.rcParams['axes.facecolor'] = 'lightgrey';
    plt.title('NB');
    plt.xlabel('Bandwith');
    plt.ylabel('Error');
    plt.plot(bandwidths, trainErrorA, '-r', label='Training error');
    plt.plot(bandwidths, trainValidA, '-k', label='Validation error');
    plt.savefig("NB", dpi=300);
    plt.show();
    '''
    return bestBandwidth;


def stats(values):
    Ys = np.array(values)[:,4];
    Xs = np.array(values)[:,:4];
    means = np.mean(Xs,0);
    stdevs = np.std(Xs,0);
    Xs = (Xs-means)/stdevs;
    return Xs,Ys;

def readFromFile(fileName):
    text = open(fileName).readlines();
    values = [];
    for lin in text:
        va = lin.split('\t');
        va[4] = va[4].split('\n')[0];
        for i, l in enumerate(va):
            va[i] = float(l);
        values.append(va);
    return values;

def getValuesFromFile(trainValues,testValues,Xs,Ys,tt, ty):
    bestBandwidth = kFolds(Ys,Xs,5,trainValues);
    global testX;
    global testY;
    testX = tt;
    testY = ty;
    score = calcB(Xs,Ys,np.array(list(enumerate(Xs)))[:,[0]].flatten(), [0],bestBandwidth,True);
    return bestBandwidth,score,best;

