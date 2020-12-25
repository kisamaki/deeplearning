import numpy as np
import copy
from collections import OrderedDict

INPUT_SIZE = 2
HIDDEN_SIZE = 20
OUTPUT_SIZE = 2

class Network:
    def __init__(self, input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, output_size = OUTPUT_SIZE):
        self.relu_mask = None
        self.affin1_x = None
        self.affin2_x = None
        
        self.params = {}
        self.params["W1"] = 0.1 * np.random.randn(input_size, hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = 0.1 * np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(output_size)
        
    
    def forword(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        B1, B2 = self.params["B1"], self.params["B2"]
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def softmax(x):
            c = np.max(x, axis=1).reshape(x.shape[0], 1)
            exp_a = np.exp(x - c)
            sum_exp_a = np.sum(exp_a, axis=1).reshape(x.shape[0], 1)
            y = exp_a / sum_exp_a
            return y
        
        def reru(x):
            return np.maximum(0, x)

        a1 = np.dot(x, W1) + B1
        z1 = reru(a1)
        a2 = np.dot(z1, W2) + B2
        result = softmax(a2)

        self.affin1_x = x
        self.relu_mask = (a1 <= 0)
        self.affin2_x = z1

        return result
    
    def backword(self, x, t):
        grads = {}
        dout = (x - t) / t.shape[0]
        grads["B2"] = np.sum(dout, axis=0)
        grads["W2"] = np.dot(self.affin2_x.T, dout)
        back_affin2 = np.dot(dout, self.params["W2"].T)
        back_affin2[self.relu_mask] = 0
        back_relu1 = back_affin2
        grads["B1"] = np.sum(back_relu1, axis=0)
        grads["W1"] = np.dot(self.affin1_x.T, back_relu1)

        return grads

    def loss(self, x, t):
        def closs_entropy(x, t):
            delta = 1e-7
            return -np.sum(t * np.log(x + delta))
        
        def mean_scuared(x, t):
            return 0.5 * np.sum((x-t)**2)
        
        y = self.forword(x)
        # result = mean_scuared(y, t)
        result = closs_entropy(y, t)
        return result
    
    def numerical_gredinet(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x)

        for i in range(x.shape[0]):
            tmp_val = copy.deepcopy(x[i])
            x[i] = tmp_val + h
            fxh = f(x)

            x[i] = tmp_val - h
            fxh2 = f(x)

            grad[i] = (fxh - fxh2) / (2*h)
            x[i] = tmp_val
        return grad
    
    def numerical_gredinet_list(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        grads["W1"] = self.numerical_gredinet(loss_W, self.params["W1"])
        grads["B1"] = self.numerical_gredinet(loss_W, self.params["B1"])
        grads["W2"] = self.numerical_gredinet(loss_W, self.params["W2"])
        grads["B2"] = self.numerical_gredinet(loss_W, self.params["B2"])

        return grads
    
if __name__ == "__main__":
    x_data = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
    t_data = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    learning_late = 0.1
    andnet = Network()

    # result = andnet.forword(x_data)
    # grads1 = andnet.backword(result, t_data)
    # grads2 = andnet.numerical_gredinet_list(x_data, t_data)
    # for key in("W1", "B1", "W2", "B2"):
    #     diff = np.average(np.abs(grads1[key] - grads2[key]))
    #     print(key + ":" + str(diff))

    for i in range(50000):
        bach_result = []
        loss = andnet.loss(x_data, t_data)
        # grads = andnet.numerical_gredinet_list(x_data, t_data)
        grads = andnet.backword(andnet.forword(x_data), t_data)
        for key in("W1", "B1", "W2", "B2"):
            andnet.params[key] -= learning_late * grads[key]
        bach_result.append(loss)
        if i % 5000 == 0:
            print("回数{}".format(i) + ": {} ".format(sum(bach_result) / len(x_data)))

    print(andnet.forword(x_data))