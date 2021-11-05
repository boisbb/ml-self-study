# importing and reading the wine data 
from src.naiveBayesClassifier import NaiveBayesClassifier
from src.utils import load_data


X_train, X_test, y_train, y_test = load_data()

model = NaiveBayesClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print ("NaiveBayesClassifier accuracy: {0:.3f}".format(model.accuracy(y_test, y_pred)))