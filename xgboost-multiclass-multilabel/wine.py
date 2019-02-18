import pandas as pd
import xgboost as xgb

from sklearn.metrics import classification_report
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X, y = load_wine(return_X_y=True)

features = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue',
    'od280/od315_of_diluted_wines', 'proline'
]

X = pd.DataFrame(data=X, columns=features)
y = pd.DataFrame(data=y, columns=['classes'])

print(X.head())
print(y.head())

print(y.classes.value_counts())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test)

params = {
    'max_depth': 6,
    'objective': 'multi:softmax',  # error evaluation for multiclass training
    'num_class': 3
}

bst = xgb.train(params, dtrain)
pred = bst.predict(dtest)

print(classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred)
print(cm)
