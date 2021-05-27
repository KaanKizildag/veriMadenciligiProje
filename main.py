from Algoritmalar import Bayesian, DecisionTreeTahmin, KNNTahmin, RandomForestTahmin
from Veri.VeriIsleme import setiBol
from Veri.reader import Reader

x_train, x_test, y_train, y_test = setiBol()


# tahminler = RandomForestTahmin.tahminEt(tmp)
# DecisionTreeTahmin.basari()
# RandomForestTahmin.basari()
# Bayesian.basari()
KNNTahmin.basari()



import datetime

tmp = datetime.datetime.now()
DecisionTreeTahmin.tahminEt(x_test[:1])
gecenSure = datetime.datetime.now() - tmp

print(f'Geçen süre:{gecenSure.microseconds} mikro saniye')