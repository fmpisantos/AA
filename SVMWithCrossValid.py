from sklearn.svm import SVC;
from ourNaive import *;

def crossValidScore(trainX,trainY,validX,validY,g):
	cl = SVC(gamma=g);
	cl.fit(trainX,trainY);
	toScoreValues = [];
	for row in range(validX.shape[0]):
		toScoreValues.append(cl.predict([validX[row]]));
	return (1-accuracy_score(np.array(validY),np.array(toScoreValues).flatten()));

def crossValidateGamma(trainValues,k):
	gammas = [];
	errors = [];
	shuffle(trainValues);
	trainXs,trainYs = stats(trainValues);
	kk = StratifiedKFold(k);
	gamma = 0.2;
	bestGamma = 0;
	bestVError = 99999;
	while gamma <=0.6:
		validError = 0;
		for train,valid in kk.split(trainYs,trainYs):
			validError = crossValidScore(trainXs[train],trainYs[train],trainXs[valid],trainYs[valid],gamma);
		if validError/k < bestVError:
			bestVError = validError/k;
			bestGamma = gamma;
		gammas.append(gamma);
		errors.append(errors);
		gamma += 0.2;
	plt.title('SVM');
	plt.xlabel('Gamma');
	plt.ylabel('Error');
	plt.plot(gammas,errors,'-r',label='Validation error');
	plt.savefig("SVM",dpi=300);
	plt.show();
	return bestGamma,bestVError;

def SVMGetScore(trainValues,testValues,gamma):
	shuffle(trainValues);
	trainXs,trainYs = stats(trainValues);
	shuffle(testValues);
	testXs,testYs = stats(testValues);
	return crossValidScore(trainXs,trainYs,testXs,testYs,gamma);