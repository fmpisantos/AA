from ourNaive import *;
from GaussianNB import *;
from SVMWithCrossValid import *;

trainValues = readFromFile("TP1_train.tsv");
testValues = readFromFile("TP1_test.tsv");
shuffle(trainValues);
shuffle(testValues);

#Our Naive Bayes
bestBandwidth,score,predicts = getValuesFromFile(trainValues,testValues);

print('Best Bandwidth:', bestBandwidth);
print('Estimate test error our Naive Bayes:', score);

#GaussianNB
GNBScore,GNBPredicts = scoreGaussian(trainValues,testValues);

print('Estimate test error Gaussian Naive Bayes:', GNBScore);

#Support Vector Machine
bestGamma , bestError = crossValidateGamma(trainValues,5);
SVMScore,SVMPredicts = SVMGetScore(trainValues,testValues,bestGamma);

print('Best Gamma:', bestGamma);
print('Estimate test error SVM:', SVMScore);

#McNemar's test
'''
print("------");
print(predicts);
print(GNBPredicts);
print(SVMPredicts);
'''