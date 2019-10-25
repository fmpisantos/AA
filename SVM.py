shuffle(trainValues);
trainXs,trainYs = stats(trainValues);

shuffle(testValues);
testXs,testYs = stats(testValues);

def crossValidScore(trainX,trainY,validX,validY,gamma):
	clf = SVC(gamma=gamma);
	clf.fit(trainX, trainY); 
	toScoreValues = [];
	for row in range(validX.shape[0]):
		toScoreValues.append(clf.predict([validX[row]]));
	return (1-accuracy_score(np.array(validY),np.array(toScoreValues).flatten()));

def crossValidateGamma(k):
    kf = StratifiedKFold(k);
    gamma = 0.2;
    bestGamma = 0;
    bestVError = 999999;
    while gamma <= 0.6:
    	tError = vError = 0;
        for train,valid in kf.split(trainYs,trainYs):
            validError = crossValidScore(trainXs[train],trainYs[train],trainXs[valid],trainYs[valid],gamma);
        if validError/k<bestVError:
            bestVError = validError/k;
            bestGamma = gamma;
        gamma+=0.2;
    return bestGamma,bestError;

def SVMGetScore(gamma):
	return crossValidScore(trainXs,trainYs,testXs,testYs,gamma);
