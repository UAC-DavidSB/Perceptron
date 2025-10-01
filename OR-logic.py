import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return math.tanh(x)

def step(x):
    return 1 if x >= 0 else 0

def linear(x):
    return x

def softmax(x_list):
    exps = [math.exp(i) for i in x_list]
    total = sum(exps)
    return [j / total for j in exps]

class Perceptron:
    def __init__(self, input_size, lr=0.1, activation='step'):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.lr = lr
        self.activation = activation

    def activate(self, x):
        if self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'relu':
            return relu(x)
        elif self.activation == 'tanh':
            return tanh(x)
        elif self.activation == 'linear':
            return linear(x)
        elif self.activation == 'step':
            return step(x)
        else:
            return step(x)

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.activate(total)

    def train(self, training_data, epochs=20):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                total_error += abs(error)

                # Actualizar pesos
                for i in range(len(self.weights)):
                    self.weights[i] += self.lr * error * inputs[i]
                self.bias += self.lr * error
            print(f"Epoch {epoch+1}, Error total: {total_error}")

# Datos de entrenamiento OR lógico
training_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1)
]

p = Perceptron(input_size=2, lr=0.1, activation='sigmoid')
p.train(training_data, epochs=20)

# Pruebas
print("Pruebas:")
for x in training_data:
    print(f"Entrada: {x[0]}, Salida esperada: {x[1]}, Predicción: {round(p.predict(x[0]), 3)}")
