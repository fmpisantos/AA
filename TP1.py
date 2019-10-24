from sklearn.neighbors.kde import KernelDensity;
from sklearn.naive_bayes import GaussianNB;
from sklearn.svm import SVC;
import numpy as np;
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold;

testX,testY = [[],[]];

def calcKDE(train,bandwith,feat):
    return KernelDensity(kernel='gaussian', bandwidth=bandwith).fit(np.array(train)[:,[feat]]);
    
def calcB(Xs,Ys, train, validation, bandwith,test):
    trainFalse = [];
    trainTrue = [];
    for i in train:
        if Ys[i] == 1.000:
            trainTrue.append(list(Xs[i]));
        else:
            trainFalse.append(list(Xs[i]));
    lnTrue = np.log(len(trainTrue)/len(train));
    lnFalse = np.log(len(trainFalse)/len(train));
    kde = [[],[]]
    print(calcKDE(trainTrue,bandwith,1));
    for f in range(4):
    	for c in range(2):
    		if(c==0):
    			kde[c].append(calcKDE(trainTrue,bandwith,f));
    		else:
    			kde[c].append(calcKDE(trainFalse,bandwith,f));
    if test:
        return (1-score(testX,testY),lnTrue,lnFalse,kde);
    else:
        return (1-score(Xs[train],Ys[train]),lnTrue,lnFalse,kde),(1-score(Xs[validation],Ys[validation],lnTrue,lnFalse,kde));
    '''
    predict = predict(train, lnTrue, lnFalse);
    toSum = calcKDE(trainTrue,validation,bandwith);
	summ = sum(toSum);
    falseSum = summ + lnFalse;
    trueSum = summ + lnTrue;
    if falseSum > trueSum:
        return falseSum;
    else:
        return trueSum;
    '''
def predict(X, pTrue, pFalse):
    #TODO: 
    
def score(X,Y, pTrue, pFalse,kde):
    return accuracy_score(Y, predict(X,pTrue, pFalse,kde));
        
#Returns Best bandwidth
def kFolds(Ys,Xs,k,values):
    kf = StratifiedKFold(k);
    bandwith = 0.02;
    bestBandwidth = 0;
    bestVError = 999999;
    while bandwith <= 0.6:
        tError, vError = 0;
        for train,valid in kf.split(Ys,Ys):
            trainError,validError = calcB(Xs,Ys,train, valid,bandwith,false);
            tError+=trainError;
            vError+=validError;
        if vError/5 < bestVError:
            bestVError = vError/5;
            bestBandwidth = bandwith;
        bandwith+=0.02;
    return bandwith;


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
def getValuesFromFile(trainFileName,testFileName):
    values = readFromFile(trainFileName);
    shuffle(values);
    Xs,Ys = stats(values);
    bestBandwidth = kFolds(Ys,Xs,5,values);
    #TODO: Ler test file e temos de fazer split sobre todo o split sobre todo o trainX e trainY, calc prob, fit e por fim score sobre o ficheiro TEST
    # Deve ficar algo assim:
    '''
    values = values = readFromFile(testFileName);
    shuffle(values);
    testX, testY = stats(values)
    score = calcB(Xs,Ys,enumerate(Xs), [0],bestBandwidth,true);
    '''
    return bestBandwidth,score;

bestBandwidth,score = getValuesFromFile("TP1_train.tsv","TP1_test.tsv");
print('Naive Bayes retrained with the complete training set using best bandwith =', best_bandwidth);
print('Estimate test error:', score);
