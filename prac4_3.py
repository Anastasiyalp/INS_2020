import numpy as np
from keras.layers import Dense
from keras.models import Sequential


def relu(x):
    return np.maximum(x, 0.)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def operation(a, b, c) -> int:
    return (a and b) or c


def get_matrix_truth():
    return np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]])


def result_of_matrix():
    return np.array([operation(*i) for i in get_matrix_truth()])


def tensor_result(dataset, weights):
    result = dataset.copy()
    layers = [relu for i in range(len(weights) - 1)]
    layers.append(sigmoid)
    for i in range(len(weights)):
        result = layers[i](np.dot(result, weights[i][0]) + weights[i][1])
    return result


def each_element_of_tensor_result(dataset, weights):
    result = dataset.copy()
    layers = [relu for i in range(len(weights) - 1)]
    layers.append(sigmoid)
    for weight in range(len(weights)):
        len_current_weight = len(weights[weight][1])
        step_result = np.zeros((len(result), len_current_weight))
        for i in range(len(result)):
            for j in range(len_current_weight):
                sum = 0
                for k in range(len(result[i])):
                    sum += result[i][k] * weights[weight][0][k][j]
                step_result[i][j] = layers[weight](sum + weights[weight][1][j])
        result = step_result
    return result


def print_predict(model, dataset):
    weights = [layer.get_weights() for layer in model.layers]
    tensor_res = tensor_result(dataset, weights)
    each_el = each_element_of_tensor_result(dataset, weights)
    model_res = model.predict(dataset)
    expr_res = [operation(data[0], data[1], data[2]) for data in dataset]
    expr = [data for data in dataset]
    assert np.isclose(tensor_res, model_res).all()
    assert np.isclose(each_el, model_res).all()
    def a(x) :
        return 1 if x > .5 else 0
    for i  in range(len(tensor_res)):
        print("Результаты вычислений для выражения -", expr[i])
        print("Тензорных выч.:             ", a(tensor_res[i][0]))
        print("Поэлементных выч.:          ", a(each_el[i][0]))
        print("Выч. через обученную модель:", a(model_res[i][0]))
        print("ОТВЕТ:                      ", expr_res[i])
        print("____________________________")

train_data = get_matrix_truth()
validation_data = result_of_matrix()

model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(3,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("ДО обучения")
print_predict(model, train_data)

model.fit(train_data, validation_data, epochs=150, batch_size=1)

print("После обучения")
print_predict(model, train_data)
