def mcClass(e0,e1,y):
	yesno = 0;
	noyes = 0;
	print(e0)
	print(e1)
	print(y)
	for i in range(y.shape[0]):
		if e0[i] == y[i] and e1[i] != y[i]:
			yesno+=1;
		if e1[i] == y[i] and e0[i] != y[i]:
			noyes += 1;
	return ((abs(yesno - noyes)-1)**2)/(yesno+noyes);
