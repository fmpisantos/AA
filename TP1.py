from ourNaive import *;
from GaussianNB import *;
from SVM import *;

trainValues = readFromFile("TP1_train.tsv");
testValues = readFromFile("TP1_test.tsv");

#Our Naive Bayes
'''
bestBandwidth,score = getValuesFromFile(trainValues,testValues);

print('Best Bandwidth:', bestBandwidth);
print('Estimate test error our Naive Bayes:', score);
'''
#GaussianNB
GNBScore = scoreGaussian(trainValues,testValues);

print('Estimate test error Gaussian Naive Bayes:', GNBScore);

#Support Vector Machine
bestGamma , bestError = crossValidateGamma(trainValues,5);
SVMScore = SVMGetScore(trainValues,testValues,bestGamma);

print('Best Gamma:', bestGamma);
print('Estimate test error our Naive Bayes:', SVMScore);