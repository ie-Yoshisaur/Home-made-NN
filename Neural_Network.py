import math
import random

class Neural_Network():
    def __init__(self, numbers_list, activation_functions_list, optimizer):
        self.layers = [Layer(numbers_list[i], numbers_list[i - 1] * int(i > 0), activation_functions_list[i], optimizer) for i in range(len(numbers_list))]

    def propagation(self, input_data):
        for i in range(len(self.layers)):
            if i == 0:
                for j in range(len(input_data)):
                    self.layers[0].neurons[j].output = input_data[j]
            else:
                for j in range(len(self.layers[i].neurons)):
                    self.layers[i].neurons[j].linear_combination(self.layers[i - 1])
                self.layers[i].activate_output()

    def back_propagation(self, training_data):
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                for j in range(len(self.layers[i].neurons)):
                    self.layers[i].neurons[j].gradient = self.layers[i].neurons[j].output - training_data[j]
            elif i != 0 and i != len(self.layers) - 1:
                self.layers[i].derivative_of_activation_function()
            elif i == 0:
                pass

            if i != 0:
                self.layers[i].propagate_gradient(self.layers[i - 1])
                for j in range(len(self.layers[i].neurons)):
                    self.layers[i].neurons[j].bias.optimize(-self.layers[i].neurons[j].gradient)
                    for k in range(len(self.layers[i].neurons[j].weights)):
                        self.layers[i].neurons[j].weights[k].optimize(self.layers[i].neurons[j].gradient * self.layers[i - 1].neurons[k].output)

class Layer():
    def __init__(self, number_of_neurons, number_of_weight, activation_function, optimizer):
        self.neurons = [Neuron(number_of_neurons, number_of_weight, activation_function, optimizer) for _ in range(number_of_neurons)]
        self.activation_function = activation_function

    def activate_output(self):
        eval(self.activation_function)(self.neurons)

    def derivative_of_activation_function(self):
        eval('derivative_of_' + self.activation_function)(self.neurons)

    def propagate_gradient(self, previous_layer):
        for i in range(len(previous_layer.neurons)):
            previous_layer.neurons[i].gradient = sum([self.neurons[j].gradient * self.neurons[j].weights[i].weight for j in range(len(self.neurons))])

class Neuron():
    def __init__(self, number_of_neurons, number_of_weight, activation_function, optimizer):
        self.output = 0
        self.weights = [Weight(box_muller_transform(1/math.sqrt(number_of_neurons) if activation_function != 'ReLU' else math.sqrt(2/number_of_neurons)), optimizer) for _ in range(number_of_weight)]
        self.bias = Weight(0, optimizer)
        self.gradient = 0

    def linear_combination(self, previous_layer):
        self.output = sum([previous_layer.neurons[i].output * self.weights[i].weight for i in range(len(self.weights))]) - self.bias.weight

class Weight():
    def __init__(self, weight, optimizer):
        self.weight = weight
        self.mean_of_gradient = 0
        self.variance_of_gradient = 0
        self.time = 0
        self.optimizer = optimizer

    def optimize(self, gradient):
        eval(self.optimizer)(self, gradient)

def box_muller_transform(variance):
    return math.sqrt(-2 * math.log(random.random())) * math.cos(2 * math.pi * random.random()) * variance

def linear(neurons):
    pass

def derivative_of_linear(neurons):
    pass

def sigmoid(neurons):
    for i in range(len(neurons)):
        neurons[i].output = 1/(1 + math.exp(-neuron[i].output))

def derivative_of_sigmoid(neurons):
    for i in range(len(neurons)):
        neurons[i].gradient = neurons[i].gradient * neurons[i].output * (1 - neurons[i].output)

def tanh(neurons):
    for i in range(len(neurons)):
        neurons[i].output = (math.exp(neurons[i].output) - math.exp(-neurons[i].output)) / (math.exp(neurons[i].output) + math.exp(-neurons[i].output))

def derivative_of_tanh(neurons):
    for i in range(len(neurons)):
        neurons[i].gradient = neurons[i].gradient * (1 - neurons[i].gradient ** 2)

def ReLU(neurons):
    for i in range(len(neurons)):
        neurons[i].output = max(neurons[i].output, 0)

def derivative_of_ReLU(neurons):
    for i in range(len(neurons)):
        neurons[i].gradient = neurons[i].gradient * int(neurons[i].output != 0)

def Softmax(neurons):
    fraction = sum([math.exp(neuron.output) for neuron in neurons])
    for i in range(len(neurons)):
        neurons[i].output = math.exp(neurons[i].output) / fraction

def SGD(weight, gradient):
    learning_rate = 0.001

    weight.weight = weight.weight - learning_rate * gradient

def Adam(weight, gradient):
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10 ** (-8)

    if gradient != 0:
        weight.time += 1
        weight.mean_of_gradient = beta1 * weight.mean_of_gradient + (1 - beta1) * gradient
        weight.variance_of_gradient = beta2 * weight.variance_of_gradient + (1 - beta2) * gradient ** 2
        hat_m = weight.mean_of_gradient/(1 - beta1 ** weight.time)
        hat_v = weight.variance_of_gradient/(1 - beta2 ** weight.time)
        weight.weight = weight.weight - learning_rate * hat_m/(math.sqrt(hat_v) + epsilon)
