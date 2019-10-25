import math;

def NormalTest(e0,y):
	error = 0;
	N = y.shape[0];
	for i in range(y.shape[0]):
		if e0[i] != y[i]:
			error+=1;
	return error,(1.96*math.sqrt(error-(error**2)/N));
