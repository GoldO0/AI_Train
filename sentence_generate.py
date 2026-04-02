import numpy as np
import os
words=["how","are","you","i","am","fine","doing","now","today","hi","hello","chatting","thanks"]
save_file="epoch_1000.npz"
def wti(word):
	index={w:i for i,w in enumerate(words)}
	return index[word]
def wtv(word):
	vect=np.zeros(len(words))
	vect[wti(word)]=1
	return vect

X=[
np.concatenate([wtv("how"),wtv("are")]),
np.concatenate([wtv("doing"),wtv("now")]),
np.concatenate([wtv("doing"),wtv("today")]),
np.concatenate([wtv("you"),wtv("are")]),
np.concatenate([wtv("hello"),wtv("hello")]),
np.concatenate([wtv("hi"),wtv("hi")])
]
Y=[
np.concatenate([wtv("fine")]),
np.concatenate([wtv("chatting")]),
np.concatenate([wtv("chatting")]),
np.concatenate([wtv("thanks")]),
np.concatenate([wtv("hi")]),
np.concatenate([wtv("hello")])
]

X=np.array(X)
Y=np.array(Y)

input_size=X.shape[1]
output_size=Y.shape[1]
learning_rate=0.01
temperature=0.7
if not os.path.exists(save_file):
	weight=np.random.randn(input_size,output_size)
	bias=np.random.randn(output_size)
	


	for epoch in range(1000):
		total_error=0
		for x,y in zip(X,Y):
			prediction=np.dot(x,weight)+bias
			error=(prediction-y)**2
			total_error+=np.sum(error)

			grad_w=np.outer(x,2*(prediction-y))
			grad_b=2*(prediction-y)

			weight-=learning_rate*grad_w
			bias-=learning_rate*grad_b
		if epoch % 100 == 0:
			print(f"Epoch: {epoch}\nTotal Error: {total_error}\n")
	np.savez("epoch_1000.npz",weight=weight,bias=bias)
else:
	data = np.load(save_file)
	weight=data["weight"]
	bias=data["bias"]

def safe_words(text):
	if text in words:
		return text
	else:
		return "hi"


def text_process(text):
	words=text
	words =words.lower()
	words=words.replace("?","").replace(".","").replace(",","").replace("!","")
	return words



def check_words(text):
	words=text_process(text).split()

	if len(words) >= 2:
		words=[words[-2],words[-1]]
	elif len(words) == 1:
		words=[words[0],words[0]]
	else:
		return ["hi","hi"]
	return words


def predict(w1,w2):
	w1=safe_words(w1)
	w2=safe_words(w2)
	encoded=np.concatenate([wtv(w1),wtv(w2)])
	prediction=np.dot(encoded,weight)+bias
	probs=np.exp(prediction/temperature) / np.sum(np.exp(prediction/temperature))
	return np.random.choice(words,p=probs)

def generate(start1,start2,length):
	w1,w2=start1,start2
	sentence=[]
	for _ in range(length):
		next_word=predict(w1,w2)
		sentence.append(next_word)
		w1,w2=w2,next_word
	return " ".join(sentence)
while True:
	inp_test=input("Enter:")
	inp_test = text_process(inp_test)
	inp_test = check_words(inp_test)
	prediction=generate(inp_test[0],inp_test[1],1)
	print("AI: ",prediction)
