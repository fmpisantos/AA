# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Assignment 1 - AA 2017
# David Moura 45235
# Rodrigo Graças 41866

import numpy as np
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score

# Get data
mat = np.genfromtxt('TP1_train.tsv', delimiter=",")

# Shuffling
data = shuffle(mat) 

# Divide into features and classes
features = data[:, :-1]
class_labels = data[:, -1]

# Standardizing
means = np.mean(features, axis=0)
devs = np.std(features, axis=0)
features = (features-means)/devs
   
# Leave 1/3 of data for test, determine parametres with cross validation with the other 2/3
X_train, X_test, Y_train, Y_test = train_test_split(features, class_labels, test_size=0.33, stratify = class_labels)

# Stratified 5-folds
folds = 5
kfolds = StratifiedKFold(Y_train, folds)


# Logistic Regression
print('\n================ Logistic Regression ================')

def calc_fold(feats, X, Y, train_ix, valid_ix, C):
    
    reg = LogisticRegression(C=C, tol=1e-10)
    reg.fit(X[train_ix, :feats], Y[train_ix])
    
    prob = reg.predict_proba(X[:, :feats])[:, 1]    
    squares = (prob-Y)**2
    
    #return the average of the training and validation error
    return np.mean(squares[train_ix]), np.mean(squares[valid_ix])


et = []
ev = []
    
C = 1 # regularization parameter
c_log = []
best_C_err = 1000000
best_C = 0

# For the regularization parameter 'C' of the logistic regression classifier
for i in range(20):
    tr_err = va_err = 0
    
    for tr_ix, va_ix in kfolds:    
        #calcular os folds sempre com a matriz toda (4 features)
        tr, v = calc_fold(4, X_train, Y_train, tr_ix, va_ix, C)
        
        tr_err += tr
        va_err += v
    
    et.append(tr_err/folds)
    ev.append(va_err/folds)
    
    if (va_err/folds) <= best_C_err:
        best_C = np.log(C)
        best_C_err = va_err/folds
    
    #print(i, ':', tr_err/folds, va_err/folds)
    
    #double C at each 20 iterations
    C = C * 2
    c_log.append(np.log(C))

#Plot the errors against the logarithm of the C value
plt.plot(c_log, et, '-b', label = "Training error")
plt.plot(c_log, ev, '-r', label = "Validation error")

plt.xlabel('Log of C value')
plt.ylabel('Mean errors')
plt.legend(loc='best')

plt.title('Logistic Regression')
plt.savefig("Logistic Regression", dpi=300)
plt.show()

print("Best regularization parameter (C):", best_C)

# Estimate true error with test set and best C value
logRegTest = LogisticRegression(C=np.exp(best_C), tol=1e-10)
logRegTest.fit(X_train, Y_train)
test_err = logRegTest.score(X_test, Y_test)
print("Estimate test error:", 1-test_err)


# K-Nearest Neighbours
print('\n================ K-Nearest Neighbours ================')

def calc_knn_error(k, X, Y, tr_ix, va_ix):
    
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X[tr_ix], Y[tr_ix])
    
    # knn.score returns the mean accuracy on the given test data and labels
    tr_score = knn.score(X[tr_ix], Y[tr_ix])
    va_score = knn.score(X[va_ix], Y[va_ix])
    
    #print("k: ", k, " tr_score: ", tr_score, " va_score: ", va_score)
    
    tr_err = 1-tr_score
    va_err = 1-va_score
    
    return tr_err, va_err


et = []
ev = []

best_mean = 1000000
best_k = 0
k_range = range(1, 40, 2) # 1 to 39 odd values


for k in k_range:    
    tr_err = va_err = 0
    
    for tr_ix, va_ix in kfolds:
        te, ve = calc_knn_error(k, X_train, Y_train, tr_ix, va_ix) 
        tr_err += te
        va_err += ve
    
    if (va_err/folds) <= best_mean:
        best_k=k
        best_mean = va_err/folds
    
    et.append(tr_err/folds)
    ev.append(va_err/folds)
        
    #print("k:", k, "et:", tr_err/folds, 'ev:', va_err/folds)

plt.plot(k_range, et, '-b', label='Training error')
plt.plot(k_range, ev, '-r', label='Validation error')
plt.xlabel('K value')
plt.ylabel('Mean errors')
plt.legend(loc='best')
plt.title('K-Nearest Neighbours')
plt.savefig("K-Nearest Neighbours", dpi=300)
plt.show()

print("Best K value:", best_k)

#Estimate true error with test set and best K value
knnTest = KNeighborsClassifier(n_neighbors = best_k) 
knnTest.fit(X_train, Y_train)
test_err_knn = knnTest.score(X_test, Y_test)

print("Estimate test error:", 1-test_err_knn)

# Naive Bayes
print('\n================ Naive Bayes ================')
prior = [[],[]]
kde = np.empty((4,2), dtype=object)
#class NaiveBayes:
    
    #def __init__(self, bandwidth, X, Y):
     #   self.split(X, Y)
      #  self.calc_prior_probability(X)
       # self.fit(bandwidth)
def split( X, Y):
        # split between real bank notes (class 0) and fake bank notes (class 1)
        mat = [[],[]]
        
        for i, elem in enumerate(X):
            #print('i:',i,' elem:', elem, 'class', Y[i])
            
            if(Y[i] == 0): # class 0
                mat[0].append(list(elem))
            
            else: # class 1
                mat[1].append(list(elem))       
        
        mat[0] = np.array(mat[0])
        mat[1] = np.array(mat[1])
        
def calc_prior_probability( X):
        # for each class, calc the log of the prior probability
        # the log of the fraction of each class in the data
        
        
        for c in range(2):
            prior[c] = np.log(float(len(mat[c])/len(X)))
            
def fit( bandwidth):  
         # dtype=object because it's stored a KDE object
        
        for feat in range(4):   # for each of the four features
            for c in range(2):  # in each of the two classes
                                # find the distribution using KDE
                                
                kde[feat][c] = KernelDensity(bandwidth).fit(mat[c][:,[feat]])
                   
def predict(X):
        prob_class_zero = np.repeat(prior[0], X.shape[0])
        prob_class_one = np.repeat(prior[1], X.shape[0])
        
        classified = []
           
        # sums all the terms in the equation...
        for feat in range(X.shape[1]): 
            prob_class_zero += kde[feat][0].score_samples(X[:,[feat]])
            prob_class_one += kde[feat][1].score_samples(X[:,[feat]])
                
        # ... and determines (classify) the class according to the maximum value found
        for i in range(X.shape[0]):
            
            if prob_class_zero[i] > prob_class_one[i]:
                classified.append(0)
            else:
               classified.append(1)
        
        return np.array(classified)
    
    
def score( X, Y):  
        
        # we used, as a measure of the accuracy, the accuracy_score function in the sklearn.metrics module
        return accuracy_score(Y,predict(X))
     
# ---------------- End of class NaiveBayes

et = []
ev = []

bandwidths = []
best_bandwidth = 0
best_error = 1000000

for bandwidth in np.arange(0.01, 1., 0.02):
    tr_err = va_err = 0
    
    for tr_ix, va_ix in kfolds:
        
    #nb = NaiveBayes(bandwidth, X_train[tr_ix], Y_train[tr_ix])
        split(X_train,Y_train)
        calc_prior_probability(X_train) 
        fit(bandwidth)
        
        te = 1 - score(X_train[tr_ix], Y_train[tr_ix])
        ve = 1 - score(X_train[va_ix], Y_train[va_ix])
        
        tr_err += te
        va_err += ve
        
    if (va_err/folds) <= best_error:
        best_error = (va_err/folds)
        best_bandwidth = bandwidth
    
    et.append(tr_err/folds)
    ev.append(va_err/folds)
    
    bandwidths.append(bandwidth)
    
    #print("bandwith:", bandwidth, "et:", tr_err/folds, 'ev:', va_err/folds)

plt.plot(bandwidths, et, '-b', label='Training error')
plt.plot(bandwidths, ev, '-r', label='Validation error')
plt.xlabel('Bandwidth')
plt.ylabel('Mean errors')
plt.legend(loc='best')
plt.title('Naive Bayes')
plt.savefig("Naive Bayes", dpi=300)
plt.show()

print("Best bandwidth:", best_bandwidth)

# Now we retrain the classifier with the complete trainning set and using the best bandwidth for the KDE
#nbTest = NaiveBayes(best_bandwidth, X_train, Y_train)
split(X_train,Y_train)
calc_prior_probability(X_train) 
fit(bandwidth)
score = score(X_test, Y_test)
#score = nbTest.score(X_test, Y_test)

print('Naive Bayes retrained with the complete training set using best bandwith =', best_bandwidth)
print('Estimate test error:', 1-score)


# ================ McNemar's Test ================
print("\n================ McNemar's Test ================")

# Compare the classifiers using McNemar's test

# c1 - first classifier
# c2 - second classifier
# e01 - number of examples the first classifier misclassifies but the second classifies correctly
# e10 - number of examples the second classifier classifies incorrectly but the first classifier classifies correctly
# return mcnemar's test with a 95% confidence interval
def mcnemar(c1, c2):
    e01 = e10 = 0
    
    for i in range(X_test.shape[0]):
        
        if c1[i] != Y_test[i] and c2[i] == Y_test[i]:
            e01 += 1
            
        elif c1[i] == Y_test[i] and c2[i] != Y_test[i]:
            e10 += 1
    
    return ((abs(e01 - e10)-1)**2)/(e01+e10)

# classifiers predictions
logReg = logRegTest.predict(X_test)
knn = knnTest.predict(X_test)
nb = score.predict(X_test)

# Logistic regression VS. K-Nearest Neighbours
print('Logistic Regression VS. K-Nearest Neighbours =', mcnemar(logReg, knn))

# K-Nearest Neighbours VS. Naive Bayes
print('K-Nearest Neighbours VS. Naive Bayes =', mcnemar(knn, nb))

# Logistic regression VS. Naive Bayes
print('Naive Bayes VS. Logistic Regression =', mcnemar(nb, logReg))