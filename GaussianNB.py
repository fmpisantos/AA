from sklearn.naive_bayes import GaussianNB;
from ourNaive import *;

def scoreGaussian(trainValues, testValues):
	shuffle(trainValues);
	trainXs,trainYs = stats(trainValues);

	shuffle(testValues);
	testXs,testYs = stats(testValues);

	clf = GaussianNB();
	clf.fit(trainXs,trainYs);

	toScoreValues = [];
	for row in range(testXs.shape[0]):
		toScoreValues.append(clf.predict([testXs[row]]));

	return (1-accuracy_score(np.array(testYs),np.array(toScoreValues).flatten()));