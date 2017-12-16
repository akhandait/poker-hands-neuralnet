import pandas as pd
import matplotlib.pyplot as plt
from helper_func import *
import numpy as np
import pickle

layers_dims = [85, 40, 40, 10]

df = pd.read_csv("train.csv")
data = df.T.values
h = data.shape[0] - 1

train_x = data[:h,:]
train_y = data[h:,:]

train_x_1 = train_x[1::2,]
train_x_2 = train_x[::2,]

train_x_onehot = np.zeros((85, 800000))

for i in range(800000):
    a = np.zeros((5, 13))
    a[np.arange(5), (train_x_1[:,i] - 1)] = 1
    a = a.flatten()
    train_x_onehot[:65,i] = a

for i in range(800000):
    a = np.zeros((5, 4))
    a[np.arange(5), (train_x_2[:,i] - 1)] = 1
    a = a.flatten()
    train_x_onehot[65:85,i] = a

train_y_onehot = np.zeros((10, train_y.shape[1]))

for i in range(train_y.shape[1]):
    train_y_onehot[train_y[0][i]][i] = 1


def predict(parameters, test_x_onehot):
    A, cache_useless = forward_propagate(test_x_onehot, parameters)
    m = test_x_onehot.shape[1]

    A = A.T
    P = np.zeros_like(A)
    P[np.arange(len(A)), A.argmax(1)] = 1
    P = P.T

    return P

def prediction_accuracy(predictions, labels):
    m = labels.shape[1]
    n = 0

    for i in range(m):
        if (predictions[:,i] == labels[:,i]).all():
            n += 1
    
    return (n / m) * 100


def model(X, Y, layers_dims, learning_rate, batch_size, test_X, test_Y, num_iterations = 3000, print_cost=False):
    costs = []                         
    parameters = initialize_parameters(layers_dims)
    m = X.shape[1]


    for i in range(num_iterations):

        for j in range(0, int(m / batch_size)):
            AL, caches = forward_propagate(X[:,j*batch_size:batch_size*(j+1)], parameters)
            #AL, caches = forward_propagate(X[:,j], parameters)

            #cost_datapoint = cost(AL, Y[:,j*batch_size:batch_size*(j+1)])
            #cost_datapoint = cost(AL, Y[:,j].reshape(10, 1))
                
            grads = backward_propagate_2(AL, Y[:,j*batch_size:batch_size*(j+1)], caches)
            #grads = backward_propagate(AL, Y[:,j*batch_size:batch_size*(j+1)], caches)
                
            parameters = update_parameters(parameters, grads, learning_rate)
        
        A_last, caches_useless = forward_propagate(X, parameters)
        cost_a = cost(A_last, Y)
        predictions = predict(parameters, test_X)
        accuracy = prediction_accuracy(predictions, test_Y)

        if print_cost and i % 1 == 0:
            print ("Cost after iteration %i: %f" %(i, cost_a))
        if print_cost and i % 1 == 0:
            print ("Accuracy after iteration %i: %f" %(i, accuracy) + ' %')
                
        if print_cost and i % 1 == 0:
            costs.append(cost_a)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

test_x_onehot = train_x_onehot[:,600000:]
test_y_onehot = train_y_onehot[:,600000:]

parameters = model(train_x_onehot[:,:800000], train_y_onehot[:,:800000], layers_dims, 0.1, 2, test_x_onehot, test_y_onehot, num_iterations = 30, print_cost = True)
fileObject = open('para.p','wb')
pickle.dump(parameters, fileObject)

predictions = predict(parameters, test_x_onehot)
accuracy = prediction_accuracy(predictions, test_y_onehot)

print(accuracy)


