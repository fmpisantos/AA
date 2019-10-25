from sklearn.svm import SVC;
from ourNaive import *;

SVMPredicts = [];

def crossValidScore(trainX,trainY,validX,validY,g):
	global SVMPredicts;
	cl = SVC(gamma=g);
	cl.fit(trainX,trainY);
	SVMPredicts = [];
	for row in range(validX.shape[0]):
		SVMPredicts.append(cl.predict([validX[row]]));
	return (1-accuracy_score(np.array(validY),np.array(SVMPredicts).flatten()));

def crossValidateGamma(trainValues,k):
	gammas = [];
	errors = [];
	shuffle(trainValues);
	trainXs,trainYs = stats(trainValues);
	kk = StratifiedKFold(k);
	gamma = 0.2;
	bestGamma = 0;
	bestVError = 99999;
	while gamma <=6:
		validError = 0;
		for train,valid in kk.split(trainYs,trainYs):
			validError = crossValidScore(trainXs[train],trainYs[train],trainXs[valid],trainYs[valid],gamma);
		gammas.append(gamma);
		errors.append(validError/k);
		if validError/k < bestVError:
			bestVError = validError/k;
			bestGamma = gamma;
		gamma += 0.2;
	plt.close();
	plt.title('SVM');
	plt.xlabel('Gamma');
	plt.ylabel('Error');
	plt.plot(gammas,errors,'-r',label='Validation error');
	plt.savefig("SVM",dpi=300);
	plt.show();
	return bestGamma,bestVError;

def SVMGetScore(trainValues,testValues,gamma):
	trainXs,trainYs = stats(trainValues);
	testXs,testYs = stats(testValues);
	return crossValidScore(trainXs,trainYs,testXs,testYs,gamma),list(np.array(SVMPredicts).flatten());