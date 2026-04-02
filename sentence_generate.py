# sentence generator 
import numpy as np 
import os
words= ["hi","there","how","are","you","i","am","fine","good","what","is","this","today","love","is","am","he","she","handsome","beautiful"]
wt_ind={w:i for i,w in enumerate(words)}
file="epoch_5000.npz"

def wtv(word):
	vect = np.zeros(len(words))
	vect[wt_ind[word]]=1
	return vect

X=[
np.concatenate([wtv("hi"),wtv("there")]), 
np.concatenate([wtv("how"),wtv("are")]),
np.concatenate([wtv("i"),wtv("am")]),
np.concatenate([wtv("what"),wtv("is")]),
np.concatenate([wtv("i"),wtv("am")]),
np.concatenate([wtv("what"),wtv("is")]),
np.concatenate([wtv("are"),wtv("you")]),
np.concatenate([wtv("what"),wtv("are")]),

np.concatenate([wtv("today"),wtv("is")]),
np.concatenate([wtv("he"),wtv("is")]),
np.concatenate([wtv("she"),wtv("is")]),
np.concatenate([wtv("you"),wtv("are")]),

np.concatenate([wtv("are"),wtv("you")]),
np.concatenate([wtv("is"),wtv("he")]),
np.concatenate([wtv("love"),wtv("is")]),
np.concatenate([wtv("this"),wtv("is")])

]

Y=[
np.concatenate([wtv("this")]),
np.concatenate([wtv("you")]),
np.concatenate([wtv("fine")]),
np.concatenate([wtv("this")]),
np.concatenate([wtv("you")]),
np.concatenate([wtv("there")]),
np.concatenate([wtv("there")]),
np.concatenate([wtv("you")]),

np.concatenate([wtv("good")]),
np.concatenate([wtv("handsome")]),
np.concatenate([wtv("beautiful")]),
np.concatenate([wtv("handsome")]),

np.concatenate([wtv("good")]),
np.concatenate([wtv("fine")]),
np.concatenate([wtv("good")]),
np.concatenate([wtv("fine")])


]

X=np.array(X)
Y=np.array(Y)
input_size=X.shape[1]
output_size=Y.shape[1]
temperature=0.7
if not os.path.exists(file):
	weight=np.random.randn(input_size,output_size)
	bias=np.random.randn(output_size)
	learning_rate=0.01

	for epoch in range(10000):
		total_error=0
		for x,y in zip(X,Y):
			predi=np.dot(x,weight)+bias
			error=(predi-y)**2
			total_error+=np.sum(error)

			grad_w=np.outer(x,2*(predi-y))
			grad_b=2*(predi-y)

			weight-=learning_rate*grad_w
			bias-=learning_rate*grad_b
		if epoch % 100 == 0:
			print("Epoch: ",epoch)
			print("Error: ",total_error)
			print()
else:
	data = np.load(file)
	weight=data["weight"]
	bias=data["bias"]

def predict(w1,w2):
	vect=np.concatenate([wtv(w1),wtv(w2)])
	pred=np.dot(vect,weight)+bias
	probs=np.exp(pred/temperature)/np.sum(np.exp(pred/temperature))
	w=np.random.choice(words,p=probs)
	return w

def generate(start1,start2,length):
	w1,w2=start1,start2
	sentence=[w1,w2]
	for _ in range(length):
		next_word = predict(w1,w2)
		sentence.append(next_word)
		w1,w2=w2,next_word 
	return " ".join(sentence)
np.savez("epoch_5000.npz",weight=weight,bias=bias)
inp_test=input("Enter:").lower().split()
ai=generate(inp_test[0],inp_test[1],30)
print(ai)

