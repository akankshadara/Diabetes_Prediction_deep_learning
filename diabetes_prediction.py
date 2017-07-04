from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# import pandas as pd

seed = 7 # fixing a random seed
np.random.seed(seed)

# loading the data set - pima_indians_diabetes.data.csv

dataset = np.genfromtxt("pima-indians-diabetes.data.csv", delimiter=',')
#splitintoinput(X)andoutput(Y)variables
X = dataset[:,0:8]
Y = dataset[:,8]

#createmodel
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
#Compilemodel
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Fitthemodel
model.fit(X, Y, nb_epoch=150, batch_size=10)
#evaluatethemodel
scores = model.evaluate(X, Y)
print("%s:%.2f%%"% (model.metrics_names[1], scores[1]*100))
