from Neural_Network import *
import csv
import random
import sys
import time

def feature_extract(input_data):
    processed_data = [0]*32
    for i in range(4):
        processed_data[int(input_data[i]) + i*8] = 1
    return processed_data

with open('dataset/iris.data') as file:
    data = file.read().split('\n')

del data[len(data) - 1]
random.shuffle(data)

number_of_variables = 4
number_of_outputs = 3
label = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
distinguish_number = 120

neural_network = Neural_Network([number_of_variables, 100, 100, 100, 100, number_of_outputs], ['linear', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], 'Adam')

print('学習を始めます。')

for i in range(distinguish_number):
    input_data = ([float(value) for value in data[i].split(',')[0:4]])
    training_data = [int(data[i].split(',')[4] == name) for name in label]
    neural_network.propagation(input_data)
    neural_network.back_propagation(training_data)
    if i%int(distinguish_number/100) == 0 or i == distinguish_number-1:
        print("\r"+'学習は'+str((i+1)*100/distinguish_number)+'%'+'完了しています。'+' '*20,end="")
        time.sleep(1)

print('学習は完了しました。')

print('検証を始めます。')

success = 0

for i in range(distinguish_number, len(data)):
    input_data = ([float(value) for value in data[i].split(',')[0:4]])
    training_data = [int(data[i].split(',')[4] == name) for name in label]
    neural_network.propagation(input_data)
    output = [neuron.output for neuron in neural_network.layers[len(neural_network.layers) - 1].neurons]
    success += int(output.index(max(output)) == training_data.index(1))
    if (i-distinguish_number)%1 == 0 or i == len(data) - 1:
        print("\r"+'検証は'+str((i+1-distinguish_number)*100/(len(data)-distinguish_number))+'%'+'完了しています。'+' '*20,end="")
        time.sleep(1)

print('学習は完了しました。')

print('この学習モデルの精度(正解率)は' + str(success*100/(len(data)-distinguish_number)) + '%' + 'です。')
