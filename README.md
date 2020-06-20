# XGBoost Multiclass & Multilabel

Here are the examples for XGboost multiclass and multilabel classification cited in the [Medium article](https://medium.com/@gabrielziegler3/multiclass-multilabel-classification-with-xgboost-66195e4d9f2d) I wrote.

## Multiclass classification tips

For multiclass, you want to set the `objective` parameter to `multi:softmax`.

> objective: `multi:softmax`: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)

Multiclass examples in `xgboost-multiclass/`

## Requirements

Install dependencies by running:

`pip install -r requirements.txt`

(You want to be using an environment to install this dependencies. If you're unsure on how to use one, follow the [docs](https://docs.python-guide.org/dev/virtualenvs/).)

## Datasets

[1] [Wine Data Set](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html): does not need to be downloaded. Can be loaded from Sklearn module using

```Python 
from sklearn.datasets import load_wine
```

[2] [Anuran Calls (MFCCs) Data Set](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29)

Download the zip folder to `datasets/`.

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20\(MFCCs\).zip -P datasets
```

Extract the zip folder so we can access `Frogs_MFCCs.csv`.

```bash
unzip datasets/Anuran\ Calls\ \(MFCCs\).zip -d datasets
```
