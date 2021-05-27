from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from DogrulukTesti.Dogruluk import dogruluguTestEt
from Veri.VeriIsleme import setiBol, veriHazirla

X_train, X_test, y_train, y_test = setiBol()

naifBayes = GaussianNB()
naifBayes.fit(X_train, y_train)

def basari():
    basliklar = np.unique(y_test)
    pred = naifBayes.predict(X_test)  # tahminleri oluşturuyorum.
    cm = confusion_matrix(pred, y_test, labels=basliklar)
    print(pd.DataFrame(data=cm, index=basliklar, columns=basliklar))

    # confusion matrisini (doğruluk matrisini göstermek için kullanıyorum)

    dogruluguTestEt(y_true=y_test, y_pred=pred)

    plot_confusion_matrix(naifBayes, X_test, y_test)
    plt.show()
    # grafik.veriyiGorsellestir(y_test,pred)

def tahminEt(X):
    # X = veriHazirla(X)
    return naifBayes.predict(X)