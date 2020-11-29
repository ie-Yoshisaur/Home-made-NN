from Neural_Network import *
import csv
import random
import sys
import time

with open('dataset/winequality-red.csv') as file:
    filedata = file.read().split('\n')

del filedata[0]
del filedata[len(filedata) - 1]
data = [[float(value) for value in row.split(';')] for row in filedata]
random.shuffle(data)

number_of_variables = 11
number_of_outputs = 10
distinguish_number = 1200

neural_network = Neural_Network([number_of_variables, 100, 100, 100, number_of_outputs], ['linear', 'ReLU', 'ReLU', 'ReLU', 'Softmax'], 'Adam')

print('学習を始めます。')

for i in range(distinguish_number):
    input_data = data[i][0:number_of_variables]
    training_data = [int(j == data[i][number_of_variables]) for j in range(number_of_outputs)]
    neural_network.propagation(input_data)
    neural_network.back_propagation(training_data)
    if i%int(distinguish_number/100) == 0 or i == distinguish_number-1:
        print("\r"+'学習は'+str((i+1)*100/distinguish_number)+'%'+'完了しています。'+' '*20,end="")
        time.sleep(1)

print('学習は完了しました。')

print('検証を始めます。')

success = 0

for i in range(distinguish_number, len(data)):
    input_data = data[i][0:number_of_variables]
    training_data = [int(j == data[i][number_of_variables]) for j in range(number_of_outputs)]
    neural_network.propagation(input_data)
    output = [neuron.output for neuron in neural_network.layers[len(neural_network.layers) - 1].neurons]
    success += int(output.index(max(output)) == training_data.index(1))
    if (i-distinguish_number)%int((len(data)-distinguish_number)/100) == 0 or i == len(data) - 1:
        print("\r"+'検証は'+str((i+1-distinguish_number)*100/(len(data)-distinguish_number))+'%'+'完了しています。'+' '*20,end="")
        time.sleep(1)

print('学習は完了しました。')

print('この学習モデルの精度(正解率)は' + str(success*100/(len(data)-distinguish_number)) + '%' + 'です。')
