from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.callbacks
import matplotlib.pyplot as plt
from var4 import gen_data


class MyCustomCallback(tensorflow.keras.callbacks.Callback):

    def __init__(self, x_train, y_train):
        super(MyCustomCallback, self).__init__()
        self.test_data = test_data
        self.test_labels = test_labels
        self.values = []

    def on_epoch_end(self, epoch, logs=None):
        count = 0
        predicted = self.model.predict(self.test_data)
        for i in range(len(predicted)):
            if abs(self.test_labels[i] - predicted[i]) >= 0.1 * abs(self.test_labels[i]):
                count += 1
        self.values.append(count)

    def on_train_end(self, logs=None):
        print(self.values)
        plt.plot(self.values)
        plt.ylabel('Number of observations with acc < 90%')
        plt.xlabel('Epochs')
        plt.show()

def loadDataImgs(length=1000, imgSize=50):
    data, labels = gen_data(length, imgSize)
    data = data.reshape(data.shape[0], imgSize, imgSize, 1)
    encoder = LabelEncoder()
    encoder.fit(labels.ravel())
    labels = encoder.transform(labels.ravel())
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3)
    return train_data, test_data, train_labels, test_labels


train_data, test_data, train_labels, test_labels = loadDataImgs()

model = Sequential()
model.add(Conv2D(32, kernel_size=(4,4), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=12, batch_size=128, validation_data=(test_data, test_labels),
          callbacks=[MyCustomCallback(test_data, test_labels)])

print("Model accuracy: %s" % (model.evaluate(test_data, test_labels)[1]))
