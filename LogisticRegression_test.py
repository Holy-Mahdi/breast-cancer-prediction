from sklearn.datasets import load_breast_cancer # add dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


data = load_breast_cancer()
X = data.data
y = data.target

feature_name = data.feature_names
target_name = data.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


lr_model = LogisticRegression(max_iter=10000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
