import pandas as pd

class Reader():
    data = None

    def getData(self):
        if self.data == None:
            self.data = pd.read_csv('Veri/Kdd99.csv')
        return self.data