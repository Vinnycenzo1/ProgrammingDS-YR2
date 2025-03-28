import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
dataframe = pd.read_csv('AirQualityPDS.csv')

array = dataframe.values
X = array[:,3:14]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = test_size, random_state = seed)

model = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
model.fit(X_train, Y_train)

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)