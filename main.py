from sklearn.datasets import load_breast_cancer # add dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


data = load_breast_cancer()
X = data.data
y = data.target

feature_name = data.feature_names
target_name = data.target_names
