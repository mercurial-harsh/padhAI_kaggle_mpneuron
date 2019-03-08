import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Class for MP Neuron
class MPNeuron:
  
    def __init__(self):
        self.b = None

    def model(self, x):
        return(sum(x) >= self.b)

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y):
        accuracy = {}

        for b in range(X.shape[1] + 1):
            self.b = b
            Y_pred = self.predict(X)
            accuracy[b] = accuracy_score(Y_pred, Y)

        best_b = max(accuracy, key = accuracy.get)
        self.b = best_b

        print('Optimal value of b is', best_b)
        print('Highest accuracy is', accuracy[best_b])

traininp = pd.read_csv("train.csv")
testdf = pd.read_csv("test.csv")

traininp['class'] = traininp['Rating'].map(lambda x:1 if x>=4 else 0)

X_raw = traininp.drop(['PhoneId','Rating','class'],axis=1)
drop_columns = ['Also Known As', 'Applications', 'Audio Features', 'Bezel-less display', 'Browser', 'Build Material', 'Co-Processor', 'Custom UI', 'Display Colour', 'Email', 'Games', 'HDMI', 'Heart Rate Monitor', 'IRIS Scanner', 'Java',                'Mobile High-Definition Link(MHL)', 'Music', 'Other Facilities', 'Phone Book', 'Ring Tone', 'Ruggedness', 'SAR Value', 'SIM 3', 'SMS', 'Screen to Body Ratio (claimed by the brand)', 'Sensor', 'Software Based Aperture', 'Special Features', 'Standby time', 'Stylus', 'TalkTime', 'USB Type-C', 'User Available Storage', 'Video Player', 'Video Recording Features', 'Waterproof', 'Wireless Charging','Screen Protection', 'Optical Image Stabilisation','Aspect Ratio','Fingerprint Sensor Position','USB OTG Support']

X_clean = X_raw.notnull().astype('int')

X = X_clean.drop(drop_columns,axis=1)

Y = traininp[['class']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify = Y, random_state=1)

X_test = X_test.values
X_train = X_train.values

mp_neuron = MPNeuron()
mp_neuron.fit(X_train, Y_train)

Y_test_pred = mp_neuron.predict(X_test)
accuracy_test = accuracy_score(Y_test_pred, Y_test)

print(accuracy_test)

X1_raw = testdf.drop(['PhoneId'],axis=1)
X1_clean = X1_raw.notnull().astype('int')
X1 = X1_clean.drop(drop_columns,axis=1)
X1 = X1.values

Y1_pred = mp_neuron.predict(X1)

finaldf = pd.concat([testdf[['PhoneId']], pd.DataFrame(Y1_pred.astype('int'))], axis=1)
finaldf.columns = ['PhoneId','Class']

finaldf.to_csv("mysubmission3.csv",index=False)