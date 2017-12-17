import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("data.csv")

output = 100
epoch = 50
time_length = 19

y = [i for i in range(100)]
x = df["上の句（ひらがな）"].values
x = [list(xx) + [0 for i in range(time_length-len(xx))] for xx in x]
test_x = df["決まり字"].values
test_x = [list(xx) + [0 for i in range(time_length-len(xx))] for xx in test_x]

word = np.array(x).reshape(-1)
lb = LabelBinarizer()
lb.fit(word)
word = lb.transform(word)
word_cls = lb.classes_
x = word.reshape(100, time_length, -1)

test_x = np.array(test_x).reshape(-1)
test_x = lb.transform(test_x)
test_x = test_x.reshape(100, time_length, -1)

lb = LabelBinarizer()
lb.fit(y)
y = lb.transform(y)

wordvec_length = x.shape[2]
nb_units = wordvec_length*3

model = Sequential()
model.add(LSTM(nb_units, batch_input_shape=(None, time_length, wordvec_length)))
model.add(Dense(output, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=epoch, validation_split=0.0)
model.save_weights('lstm.hdf5')

model.load_weights('lstm.hdf5')
pred = model.predict(test_x)
pred_class  = np.argmax(pred, axis = 1)
tag_class  = np.argmax(y, axis = 1)

pd.concat((pd.DataFrame(pred_class, columns=["pred"]), pd.DataFrame(tag_class, columns=["true"])), axis=1).to_csv("result.csv", index=False)

test_acc = accuracy_score(pred_class, tag_class)
print(test_acc)