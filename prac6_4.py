from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from var4 import gen_data


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
model.fit(train_data, train_labels, epochs=3, batch_size=10, validation_data=(test_data, test_labels))

print("Model accuracy: %s" % (model.evaluate(test_data, test_labels)[1]))