from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

from Veri.reader import Reader


def donustur(data, index):
    oneHotEncoder = preprocessing.OneHotEncoder()
    # kategorik veriden sayısal bir ifadeye geçmek için one hot encoder kullanıyorum
    tmp = data.iloc[:, index].values  # ilgili kolonu alıyorum


    tmp = oneHotEncoder.fit_transform(tmp.reshape((-1, 1))).toarray()
    columnNames = oneHotEncoder.get_feature_names()
    tmp = pd.DataFrame(data=tmp, columns=columnNames)
    return pd.concat([tmp, data.drop(columns=[data.columns[index]])], axis=1)


def veriHazirla(X):
    try:
        X = donustur(X, X.columns.get_loc('service'))
        X = donustur(X, X.columns.get_loc('protocol_type'))
        X = donustur(X, X.columns.get_loc('flag'))
    except KeyError:
        pass
    return X
def setiBol():
    datas = Reader().getData()
    Y = datas['outcome'].values
    X = datas.drop(columns=['outcome','id'])
    X = veriHazirla(X)
    return train_test_split(X, Y, test_size = 0.33, random_state=42)