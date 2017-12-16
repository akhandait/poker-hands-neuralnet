import numpy as np

def normalize_data(x):
    m = x.shape[1]    # total number of examples
    mean = np.mean(x, axis = 0).reshape((1, m))
    std = np.std(x, axis = 0).reshape((1, m))
    
    return (x - mean) / std    # broadcasting


def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims)
    
    for i in range(1, L):
        # He initialization
        parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1]) * (2 / np.sqrt(layers_dims[i-1]))
        parameters['b' + str(i)] = np.ones((layers_dims[i], 1))

    return parameters


def sigmoid_forward(z):
    return 1 / (1 + np.exp(-z)), z


def relu_forward(z):
    activation_cache = z
    z[z<0] = 0 
    
    return z, activation_cache


def softmax_forward(z):
    exps = np.exp(z)
    activation_cache = z

    return exps / np.sum(exps, axis = 0, keepdims = True), activation_cache


def linear_forward(A, W, b):
    z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    
    return z, linear_cache


def linear_activation_forward(A_prev, W, b, activation):
    z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid":
        A, activation_cache = sigmoid_forward(z)
        
    elif activation == "softmax":
        A, activation_cache = softmax_forward(z)
        
    cache = (linear_cache, activation_cache)
    
    return A, cache


def forward_propagate(x, parameters):
    caches = []

    A = x
    """
    A = np.copy(x)
    if A.size == 85:
        A = A.reshape((85, 1))
    """
    
    L = len(parameters) // 2

    for i in range(1, L):
        A_prev = np.copy(A)
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(i)], parameters['b' + str(i)], activation = "sigmoid")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(i+1)], parameters['b' + str(i+1)], activation = "softmax")
    caches.append(cache)
    
    return AL, caches


def cost(AL, Y):
    m = Y.shape[1]
    t1 = np.sum(np.sum(Y * np.log(AL), axis = 0))
    return - t1 / m


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) 
    db = np.sum(dZ, axis = 1, keepdims = True) 
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def sigmoid_backward(dA, activation_cache):
    s = 1 / (1 + np.exp(-activation_cache))
    sigmoid_derivative = s * (1 - s)
    return dA * sigmoid_derivative

def sigmoid_derivative(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s)

def relu_backward(dA, activation_cache):
    activation_cache[activation_cache >= 0] = 1    # activating the given Z
    activation_cache[activation_cache < 0] = 0    # taking the derivative of relu in the same array
    
    return dA * activation_cache


def softmax_backward(dA, activation_cache):
    exps = np.exp(activation_cache)
    softmax = exps / np.sum(exps, axis = 0, keepdims = True)
    softmax_derivative = [x * (1 - x) for x in softmax]
    
    return dA * softmax_derivative


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    if activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db    


def backward_propagate(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL =  - (Y / AL)
    
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "softmax")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, 'sigmoid')
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads    


def backward_propagate_2(AL, Y, caches):
    grads = {}

    d3 = AL - Y
    d2 = sigmoid_derivative(caches[1][1]) * np.dot(caches[2][0][1].T, d3)
    d1 = sigmoid_derivative(caches[0][1]) * np.dot(caches[1][0][1].T, d2)

    grads['dW3'] = np.dot(d3, caches[2][0][0].T)
    grads['db3'] = np.sum(d3, axis = 1, keepdims = True)
    grads['dW2'] = np.dot(d2, caches[1][0][0].T)
    grads['db2'] = np.sum(d1, axis = 1, keepdims = True)
    grads['dW1'] = np.dot(d1, caches[0][0][0].T)
    grads['db1'] = np.sum(d1, axis = 1, keepdims = True)

    return grads
    

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for i in range(L):
        parameters['W' + str(i+1)] = parameters['W' + str(i+1)] - learning_rate * grads["dW" + str(i + 1)]
        parameters['b' + str(i+1)] = parameters['b' + str(i+1)] - learning_rate * grads["db" + str(i + 1)]
        
    return parameters    
    
