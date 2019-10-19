from sklearn.neighbors.kde import KernelDensity;
from sklearn.naive_bayes import GaussianNB;
from sklearn.svm import SVC;

def getValuesFromFile(fileName):
    text = open(fileName).readlines();
    values = [];
    for lin in text:
        va = lin.split('\t');
        va[4] = va[4].split('\n')[0];
        values.append(va);
    print(values);
    return values;

getValuesFromFile("TP1_train.tsv");
getValuesFromFile("TP1_test.tsv");