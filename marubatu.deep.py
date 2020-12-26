import numpy as np
import random
import copy
from collections import OrderedDict
import marubatu 

INPUT_SIZE = 9
HIDDEN_SIZE = 10
OUTPUT_SIZE = 9
learning_late = 0.1

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
        # result = softmax(a2)

        self.affin1_x = x
        self.relu_mask = (a1 <= 0)
        self.affin2_x = z1

        return a2
    
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

    def loss(self, state, action, reword, next_state):
        init_q = self.forword(state)
        init_qs = []
        i = 0
        for data in init_q:
            init_qs.append(data[action[i]])
            i += 1

        init_q = np.array(init_qs)
        x = self.forword(next_state)
        next_q_max = [np.max(x) for x in x]
        result = init_q - (reword + next_q_max)
        return sum(result ** 2)
    
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
    
    def numerical_gredinet_list(self, state, action, reword, next_state):
        loss_W = lambda W: self.loss(state, action, reword, next_state)

        grads = {}

        grads["W1"] = self.numerical_gredinet(loss_W, self.params["W1"])
        grads["B1"] = self.numerical_gredinet(loss_W, self.params["B1"])
        grads["W2"] = self.numerical_gredinet(loss_W, self.params["W2"])
        grads["B2"] = self.numerical_gredinet(loss_W, self.params["B2"])

        return grads

if __name__ == "__main__":
    agent = Network()

    x_data = np.array([[1, 1, 0, 2, 0, 0, 0, 0, 2], [2, 1, 2, 0, 1, 0, 0, 0, 0]])
    t_data = np.array([[-1, -1, 1, -1, 0, 0, 0, 0, -1], [-1, -1, -1, 0, -1, 0, 0, 1, 0]])

    episode = 5
    bach_size = 100
    epoch = 50

    # result = agent.forword(x_data)
    # grads1 = agent.backword(result, t_data)
    # grads2 = agent.numerical_gredinet_list(x_data, t_data)
    # for key in("W1", "B1", "W2", "B2"):
    #     diff = np.average(np.abs(grads1[key] - grads2[key]))
    #     print(key + ":" + str(diff))

    for genarate in range(episode):
        episode_memory = []
        for i in range(bach_size):
            game = marubatu.OXgame()
            while game.continue_ == True:
                memory = {}
                memory["state"] = copy.deepcopy(game.BOARD)
                memory["action"] = np.argmax(agent.forword(np.array(game.BOARD)))
                reword = game.game_step(memory["action"])
                memory["reword"] = reword
                memory["next_state"] = copy.deepcopy(game.BOARD)
                episode_memory.append(memory)
        
        x_data = random.sample(episode_memory, bach_size)
        x_state = np.array([x.get("state") for x in x_data])
        x_action = np.array([x.get("action") for x in x_data])
        x_reword = np.array([x.get("reword") for x in x_data])
        x_next_state = np.array([x.get("next_state") for x in x_data])

        if genarate == 0:
            print(agent.loss(x_state, x_action, x_reword, x_next_state))

        for _ in range(epoch):
            grads = agent.numerical_gredinet_list(x_state, x_action, x_reword, x_next_state)
            for key in("W1", "B1", "W2", "B2"):
                agent.params[key] -= learning_late * grads[key]
        
        reword_list = []
        for _ in range(100):
            game = marubatu.OXgame()
            while game.continue_ == True:
                action = np.argmax(agent.forword(np.array(game.BOARD)))
                reword = game.game_step(action)
            reword_list.append(reword)

        win_ = reword_list.count(1)
        defate_ = reword_list.count(-1)
        draw_ = reword_list.count(0)

        print(agent.loss(x_state, x_action, x_reword, x_next_state))
        print("勝利数: {}".format(win_))
        print("敗北数: {}".format(defate_))
        print("引き分け数: {}".format(draw_))
            

    #     for i in range(1000):
    #         bach_result = []
    #         loss = agent.loss(x_data, t_data)
    #         # grads = agent.numerical_gredinet_list(x_data, t_data)
    #         grads = agent.backword(agent.forword(x_data), t_data)
    #         for key in("W1", "B1", "W2", "B2"):
    #             agent.params[key] -= learning_late * grads[key]
    #         bach_result.append(loss)
    #         if i % 10 == 0:
    #             print("回数{}".format(i) + ": {} ".format(sum(bach_result) / len(x_data)))

    # print(agent.forword(x_data))