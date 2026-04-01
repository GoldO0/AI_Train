import numpy as np 

lets=["a","b","c"]

def ltv(let):
	vect=np.zeros(3)
	vect[lets.index(let)]=1
	return vect

X=np.array([ltv("a"),ltv("b")])
Y=np.array([ltv("b"),ltv("c")])

weight=np.random.randn(3,3)
bias=np.random.randn(3)
learning_rate=0.01

for epoch in range(1000):
	total_error=0
	for x,y in zip(X,Y):

		prediction=np.dot(x,weight)+bias
		error=(prediction-y)**2
		total_error+=sum(error)

		grad_w=np.outer(x,2*(prediction-y))
		grad_b=2*(prediction-y)
		weight-=learning_rate*grad_w
		bias-=learning_rate*grad_b
	print("Error: ",total_error)

let = input()
let_to_vect=ltv(let)
predict=np.dot(let_to_vect,weight)+bias
word=lets[np.argmax(predict)]
print(word)