import numpy as np
import random
import copy
from collections import OrderedDict
import marubatu
import csv

INPUT_SIZE = 9
HIDDEN_SIZE = 20
OUTPUT_SIZE = 9
EPSOLON = 0.1
GAMMA = 0.9
learning_late = 0.1
0
def write_csv(file, save_dict):
    with open(file, "w") as f:
        w = csv.writer(f)
        save_row = []
        for key, values in save_dict.items():
            a = []
            a.append(key)
            if values.ndim != 1:
                for value in values:
                    a.append(value)
            else:
                a.append(values)
            save_row.append(a)
        w.writerows(save_row)

def read_dict(file):
    return_dict = {}
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            return_dict[row[0]] = []
            for value in row:
                if value == row[0]:
                    continue
                value = value.strip("[").strip("]").strip(" ").split(" ")
                value = [float(x) for x in value if x != ""]
                return_dict[row[0]].append(value)
        
    return return_dict

class Network:
    def __init__(self, input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, output_size = OUTPUT_SIZE, learning_late = learning_late):
        self.relu_mask = None
        self.affin1_x = None
        self.affin2_x = None
        self.learning_late = learning_late
        
        self.params = {}
        self.params["W1"] = 0.1 * np.random.randn(input_size, hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = 0.1 * np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(output_size)
        
    
    def forword(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        B1, B2 = self.params["B1"], self.params["B2"]
        # def sigmoid(x):
        #     return 1 / (1 + np.exp(-x))

        # def softmax(x):
        #     c = np.max(x, axis=1).reshape(x.shape[0], 1)
        #     exp_a = np.exp(x - c)
        #     sum_exp_a = np.sum(exp_a, axis=1).reshape(x.shape[0], 1)
        #     y = exp_a / sum_exp_a
        #     return y
        
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
    
    def backword(self, y, t):
        grads = {}
        dout = (y - t) / t.shape[0]
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
        result = mean_scuared(y, t)
        # result = closs_entropy(y, t)
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


class Agent:
    def __init__(self, actions=OUTPUT_SIZE, gamma = GAMMA):
        self.main_net = Network()
        self.target_net = Network()
        self.epsilon = EPSOLON
        self.actions = actions
        self.gamma = gamma
        self.memory = {}
    
    def act(self, x):
        if np.random.randint(0, 100) < self.epsilon:
            action = np.random.randint(0, self.actions)
        else:
            action = np.argmax(self.main_net.forword(x))
        return action
    
    def learn(self, x, t):
        y = self.main_net.forword(x)
        loss = self.main_net.loss(x, t)
        grads = self.main_net.backword(y, t)
        for key in("W1", "B1", "W2", "B2"):
            self.main_net.params[key] -= learning_late * grads[key]
        
        return loss

    def target_update(self):
        self.target_net.params = copy.deepcopy(self.main_net.params)


def model_learn(episode, batch_size):
    agent = Agent()
    min_loss = 100
    max_genarate = None
    max_ganerate_params = None

    episode = episode
    batch_size = batch_size
    for genarate in range(episode):
        episode_memory = []
        for i in range(batch_size):
            game = marubatu.OXgame()
            while game.continue_ == True:
                memory = {}
                memory["state"] = copy.deepcopy(game.BOARD)
                memory["action"] = agent.act(np.array(game.BOARD))
                reword = game.game_step(memory["action"], 100)
                memory["reword"] = reword
                if game.continue_ == True:
                    memory["next_state"] = copy.deepcopy(game.BOARD)
                else:
                    memory["next_state"] = None
                episode_memory.append(memory)
        
        learn_data_index = np.random.randint(0, len(episode_memory), len(episode_memory))

        x_data = []
        t_data = []
        for i in learn_data_index:
            learn_data_x = episode_memory[i]["state"]
            learn_data_t = agent.main_net.forword(episode_memory[i]["state"])
            if episode_memory[i]["next_state"] == None:
                learn_data_t[episode_memory[i]["action"]] = episode_memory[i]["reword"]
            else:
                learn_data_t[episode_memory[i]["action"]] = episode_memory[i]["reword"] + agent.gamma * np.max(agent.target_net.forword(episode_memory[i]["next_state"]))
            x_data.append(learn_data_x)
            t_data.append(learn_data_t)
        x_data = np.array(x_data)
        t_data = np.array(t_data)

        loss = agent.learn(x_data, t_data)
        if genarate >= episode - 100 and loss <= min_loss:
                min_loss = loss
                max_genarate = genarate
                max_ganerate_params = agent.main_net.params

        #勾配確認
        # result = agent.main_net.forword(x_data)
        # grads1 = agent.main_net.backword(result, t_data)
        # grads2 = agent.main_net.numerical_gredinet_list(x_data, t_data)
        # for key in("W1", "B1", "W2", "B2"):
        #     diff = np.average(np.abs(grads1[key] - grads2[key]))
        #     print(key + ":" + str(diff))

        if (genarate + 1) % 100 == 0:
            agent.target_update()
            reword_list = []
            for _ in range(100):
                game = marubatu.OXgame()
                while game.continue_ == True:
                    action = np.argmax(agent.main_net.forword(game.BOARD))
                    reword = game.game_step(action, 70)
                reword_list.append(reword)

            win_ = reword_list.count(1)
            defate_ = reword_list.count(-1)
            draw_ = reword_list.count(0)

            print("世代: {}".format(genarate+1))
            print(loss)
            print("勝利数: {}".format(win_))
            print("敗北数: {}".format(defate_))
            print("引き分け数: {}".format(draw_))

    reword_list = []
    for _ in range(100):
        game = marubatu.OXgame()
        while game.continue_ == True:
            action = np.argmax(agent.main_net.forword(game.BOARD))
            reword = game.game_step(action, 100)
        reword_list.append(reword)

    win_ = reword_list.count(1)
    defate_ = reword_list.count(-1)
    draw_ = reword_list.count(0)

    print("最高世代: {}".format(max_genarate+1))
    print(min_loss)
    print("勝利数: {}".format(win_))
    print("敗北数: {}".format(defate_))
    print("引き分け数: {}".format(draw_))
    write_csv("OX_DQN.csv", max_ganerate_params)

def model_play():
    params = read_dict("OX_DQN.csv")
    for key in params.keys():
        params[key] = np.array(params[key])
    game = marubatu.OXgame()
    OXagent = Agent()
    OXagent.main_net.params = params

    while(game.judge(game.BOARD) == 0):
            game.drawborad()
            if game.setstoneO(np.argmax(OXagent.main_net.forword(game.BOARD))) == False:
                game.drawborad()
                print("X win")
                break
            else:
                judge_ = game.judge(game.BOARD)
                if judge_ == 1:
                    game.drawborad()
                    print("O win")
                    break
                elif judge_ == 3:
                    game.drawborad()
                    print("draw")
                    break
            game.drawborad()
            if game.setstoneX(int(input("座標を入力してください: "))) == False:
                game.drawborad()
                print("O win")
                break
            else:
                judge_ = game.judge(game.BOARD)
                if judge_ == 2:
                    game.drawborad()
                    print("X win")
                    break
                elif judge_ == 3:
                    game.drawborad()
                    print("draw")
                    break

if __name__ == "__main__":
    episode = 20000
    batch_size = 10

    model_learn(episode, batch_size)
    model_play()
    


