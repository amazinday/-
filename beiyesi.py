import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score

iris_data = pd.read_csv("iris.csv")
data = iris_data[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].values
label = iris_data['Species'].values
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_pred, y_test))
