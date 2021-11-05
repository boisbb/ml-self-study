from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from src.utils import load_data


X_train, X_test, y_train, y_test = load_data()

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Scikit-learn GaussianNB accuracy: {0:.3f}".format(accuracy_score(y_test, y_pred)))
