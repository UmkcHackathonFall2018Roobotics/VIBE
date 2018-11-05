# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
import numpy
import pandas as pd
from keras.layers import Dense
from IPython.display import SVG
from keras.utils import plot_model
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
 
# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=3, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("...model created")
    
    return model
 
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
data = pd.read_csv("sentiment_stocks.csv")

# split into input (X) and output (Y) variables
X = data.iloc[:,0:3]
Y = data.iloc[:,3]

kfold = StratifiedKFold(n_splits=2)
cvscores = []

for train, test in kfold.split(X, Y):
    # create model
    model = create_model()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train, X_test = X.iloc[train], X.iloc[test]
    Y_train, Y_test = Y.iloc[train], Y.iloc[test]
    # Fit the model
    model.fit(X.iloc[train], Y.iloc[train], epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))