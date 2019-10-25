from ourNaive import *;
from GaussianNB import *;
from SVMWithCrossValid import *;
from Mc import *;
from NormalTest import *;
import os;

trainValues = readFromFile("TP1_train.tsv");
testValues = readFromFile("TP1_test.tsv");
trainValues = shuffle(trainValues);
testValues = shuffle(testValues);
Xs,Ys = stats(trainValues);
testX, testY = stats(testValues);

#Our Naive Bayes
bestBandwidth,score,predicts = getValuesFromFile(trainValues,testValues,Xs,Ys,testX, testY);

print('Best Bandwidth:', bestBandwidth);
print('Estimate test error our Naive Bayes:', score);

#GaussianNB
GNBScore,GNBPredicts = scoreGaussian(Xs,Ys,testX, testY);

print('Estimate test error Gaussian Naive Bayes:', GNBScore);
#Support Vector Machine
bestGamma , bestError = crossValidateGamma(trainValues,5,Xs,Ys,testX, testY);
SVMScore,SVMPredicts = SVMGetScore(bestGamma,Xs,Ys,testX, testY);

print('Best Gamma:', bestGamma);
print('Estimate test error SVM:', SVMScore);

#McNemar's test
MCNBvsGNB = mcClass(predicts,GNBPredicts, testY);
MCNBvsSVM = mcClass(predicts,SVMPredicts, testY);
SVMvsGNB = mcClass(SVMPredicts,GNBPredicts, testY);

print('Our NB vs Gaussian NB McNemars test:', MCNBvsGNB);
print('Our NB vs SVM McNemars test:', MCNBvsSVM);
print('SVM vs Gaussian NB McNemars test:', SVMvsGNB);

#Normal test
MCNBErrors,normalMCNB = NormalTest(predicts,testY);
GNBErrors,normalGNB = NormalTest(GNBPredicts,testY);
SVMErrors,normalSVM = NormalTest(SVMPredicts,testY);

print('Our NB Normal test:'+ str(MCNBErrors)+"+-"+str(normalMCNB));
print('Gaussian NB Normal test:'+ str(GNBErrors)+"+-"+str(normalGNB));
print('SVM Normal test:'+ str(SVMErrors)+"+-"+str(normalSVM));