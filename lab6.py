import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import imdb

TEST_DIMENSIONS = 10000;


def vectorize(sequences, dimension=TEST_DIMENSIONS):
    result = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        result[i, sequence] = 1
    return result


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=TEST_DIMENSIONS)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)


data = vectorize(data, )
targets = np.array(targets).astype("float32")

test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

model = Sequential()

# Input - Layer
model.add(Dense(50, activation='relu', input_shape=(TEST_DIMENSIONS, )))

# Hidden - Layers
model.add(Dropout(0.3, noise_shape=None, seed=None))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.2, noise_shape=None, seed=None))
model.add(Dense(50, activation="relu"))

# Output- Layer
model.add(Dense(1, activation="sigmoid"));
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]);

H = model.fit( train_x, train_y, epochs=2, batch_size = 500, validation_data = (test_x, test_y))


plt.plot(H.history['loss'], 'b', label='train')
plt.plot(H.history['val_loss'], 'r', label='validation')
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()

plt.plot(H.history['accuracy'], 'b', label='train')
plt.plot(H.history['val_accuracy'], 'r', label='validation')
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()


custom_x = [
        "There is something much more in this movie than meets the eye. Watch it and find for yourself",
        "I had the privilege of seeing this movie before it came out, it blew me away.",
        "Bottom Line, this is a movie i could write a novel about, and it certainly deserves that attention.",
        "This movie is a disaster on many levels, but where it fails most miserably is at attempting to put dreams on screen.",
        "Okay, maybe this movie isn't awful."
]
custom_y = [1., 1., 1., 0., 0.]


def gen_custom_x(custom_x, word_index):
    def get_index(a, index):
        new_list = a.split()
        for i, v in enumerate(new_list):
            ind = index.get(v.lower());
            new_list[i] = ind
        return new_list
    for i in range(len(custom_x)):
        custom_x[i] = get_index(custom_x[i], word_index)
    for index_j, i in enumerate(custom_x):
        for index, value in enumerate(i):
            if value is None:
                custom_x[index_j][index] = 0
            elif value > 10000:
                custom_x[index_j][index] = 0
    return custom_x


print('Before: {}'.format(custom_x))
custom_x = gen_custom_x(custom_x, imdb.get_word_index())
print('After: {}'.format(custom_x))


custom_y = np.asarray(custom_y).astype("float32")
custom_x = vectorize(custom_x)
print(custom_x, custom_y)


_, acc = model.evaluate(custom_x, custom_y)
preds = model.predict(custom_x)
print('acc:', acc)
print(preds)
plt.title("dataset predications")
plt.plot(custom_y, 'r', label='real')
plt.plot(preds, 'b', label='pred')
plt.legend()
plt.show()
plt.clf()
