# learn 3x-10

import numpy as np 

X=np.array([1,2,3,4,5],dtype=float)
Y=np.array([-7,-4,-1,2,5],dtype=float)

weight=np.random.randn()
bias=np.random.randn()

learning_rate=0.01

for epoch in range(10000):
	total_error=0
	for x,y in zip(X,Y):
		prediction=weight*x + bias

		error=(prediction-y)**2
		total_error+=error

		gradientW=2*(prediction-y)*x
		gradientB=2*(prediction-y)

		weight-= learning_rate*gradientW
		bias-=learning_rate*gradientB

	if epoch % 100 == 0:
		print(f"Error:{total_error}\n")
while True:
	num=int(input("Enter num:"))
	predict=weight*num+bias
	print(f"Prediction: {predict}")