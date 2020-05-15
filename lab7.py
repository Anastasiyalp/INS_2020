import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D
from keras_preprocessing import sequence
from keras.datasets import imdb


TEST_DIMENSIONS = 10000
max_review_length = 500

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=TEST_DIMENSIONS)
training_data = sequence.pad_sequences(training_data, maxlen=max_review_length)
testing_data = sequence.pad_sequences(testing_data, maxlen=max_review_length)

data = np.concatenate((training_data, testing_data))
targets = np.concatenate((training_targets, testing_targets))
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]

embedding_vecor_length = 32


def build_model_1():
    model = Sequential()
    model.add(Embedding(TEST_DIMENSIONS, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def build_model_2():
    model = Sequential()
    model.add(Embedding(TEST_DIMENSIONS, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def plot_result(H, name):
    plt.plot(H.history['loss'], 'b', label='train')
    plt.plot(H.history['val_loss'], 'r', label='validation')
    plt.title('Loss' + name)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    plt.clf()

    plt.plot(H.history['accuracy'], 'b', label='train')
    plt.plot(H.history['val_accuracy'], 'r', label='validation')
    plt.title('accuracy' + name)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    plt.clf()


def ensembling_models(model1, model2):
    predictions1 = model1.predict(testing_data)
    predictions2 = model2.predict(testing_data)
    predictions = np.divide(np.add(predictions1, predictions2), 2)
    predictions = np.greater_equal(predictions, np.array([0.5]))
    predictions = np.logical_not(np.logical_xor(predictions, testing_targets))
    acc = predictions.mean()
    print("Accuracy of ensembling models is %s" % acc)


model1 = build_model_1()
model2 = build_model_2()
H1 = build_model_1().fit(train_x, train_y, validation_data=(test_x, test_y), epochs=3, batch_size=64)
H2 = build_model_2().fit(train_x, train_y, validation_data=(test_x, test_y), epochs=3, batch_size=64)

scores = model1.evaluate(testing_data, testing_targets, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
plot_result(H1, " model1")
scores = model2.evaluate(testing_data, testing_targets, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
plot_result(H2, " model2")

predictions1 = model1.predict(testing_data)
predictions2 = model2.predict(testing_data)
predictions = np.divide(np.add(predictions1, predictions2), 2)
predictions = np.greater_equal(predictions, np.array([0.5]))
predictions = np.logical_not(np.logical_xor(predictions, testing_targets))
acc = predictions.mean()
print("Accuracy of ensembling models is %s" % acc)


custom_x = [
        "Watch it and find for yourself",
        "I had the privilege of seeing this movie before it came out, it blew me away.",
        "it certainly deserves that attention.",
        "This movie is a disaster on many levels.",
        "this movie great."
]
custom_y = [1., 1., 1., 0., 1.]


def gen_custom_x(custom_x, word_index):
    def get_index(a, index):
        new_list = a.split()
        for i, v in enumerate(new_list):
            new_list[i] = index.get(v.lower())
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


custom_x = sequence.pad_sequences(gen_custom_x(custom_x, imdb.get_word_index()), maxlen=max_review_length)
predictions1 = model1.predict(custom_x)
predictions2 = model2.predict(custom_x)
predictions = np.divide(np.add(predictions1, predictions2), 2)

print('acc:', acc)
print(predictions)
plt.title("dataset predications")
plt.plot(custom_y, 'r', label='real')
plt.plot(predictions, 'b', label='pred')
plt.legend()
plt.show()
plt.clf()

predictions = np.greater_equal(predictions, np.array([0.5]))
predictions = np.logical_not(np.logical_xor(predictions, custom_y))
_, acc1 = model1.evaluate(custom_x, custom_y)
_, acc2 = model2.evaluate(custom_x, custom_y)
print("Validation accuracy of 1st model is %s" % acc1)
print("Validation accuracy of 2nd model is %s" % acc2)
acc = predictions.mean()
print("Validation accuracy of ensembling models is %s" % acc)
