import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess(train, test, target="Heart Disease"):
    
    X = train.drop(columns=[target])
    y = train[target]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = pd.DataFrame(X)
    y = pd.Series(y)

    feature_names = list(X.columns)

    return X, y, test, feature_names, le