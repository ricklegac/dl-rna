import numpy as np

class MPNeuron:

	def __init__(self):
		self.thershold = None

	def model(self,x): #x es un array
		z=sum(x)
		return (z>=self.thershold) #retorna true si es mayor al threshold

	def predict(self,X):
		
		Y=[]
		for x in X:
			result = self.model(x)
			Y.append(result)

		return np.array(Y)


mp_neuron = MPNeuron()
mp_neuron.thershold = 3
predict=mp_neuron.predict([[1,0,1,0],[1,1,1,1],[1,0,1,1]])
print(predict)


