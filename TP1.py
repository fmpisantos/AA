from ourNaive import getValuesFromFile;

bestBandwidth,scoree = getValuesFromFile("TP1_train.tsv","TP1_test.tsv");
print('Naive Bayes retrained with the complete training set using best bandwith =', bestBandwidth);
print('Estimate test error:', scoree);
