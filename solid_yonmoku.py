import random

class OXgame:
    def __init__(self):
        self.BOARD = [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]
                    ]
        self.continue_ = True

    def drawborad(self):
        for i in range(len(self.BOARD)):
            if self.BOARD[i] == 0:
                print(str(i), end="")
            elif self.BOARD[i] == 1:
                print("O", end="")
            elif self.BOARD[i] == 2:
                print("X", end="")
            if (i+1) % 3 == 0:
                print()
        print("**********")

    def setstoneO(self, i):
        set_line = self.BOARD[i]
        try:
            index = set_line.index(0)
            self.BOARD[i][index] = 1
            self.judge(i, index)
        except:
            return False
            
        
    def setstoneX(self, i):
        set_line = self.BOARD[i]
        try:
            index = set_line.index(0)
            self.BOARD[i][index] = 2
            self.judge(i, index)
        except:
            return False

    def judge(self, i, index):
        player = self.BOARD[i][index]
        if self.BOARD[i].count(player):
            return player
        for i in []

        
    
    def cpu(self, persent):
        select_list = []
        return_ = None
        BOARD_copy = self.BOARD.copy()
        for i in range(len(self.BOARD)):
            if self.BOARD[i] == 0:
                select_list.append(i)
        return_ = select_list[random.randrange(len(select_list))]
        if random.randint(0, 100) < persent:
            for i in select_list:
                BOARD_copy[i] = 1
                if self.judge(BOARD_copy) == 1:
                    return_ = i
                BOARD_copy[i] = 0
            for i in select_list:
                BOARD_copy[i] = 2
                if self.judge(BOARD_copy) == 2:
                    return_ = i
                BOARD_copy[i] = 0
        
        return return_
    
    def game_step(self, action, cpu_persent = 100):
        self.continue_ = False
        if self.setstoneO(action) == False:
                return -1
        else:
            judge_ = self.judge(self.BOARD)
            if judge_ == 1:
                return 1
            elif judge_ == 3:
                return 0
            if self.setstoneX(self.cpu(cpu_persent)) == True:
                return 1
            else:
                judge_ = self.judge(self.BOARD)
                if judge_ == 2:
                    return -1
                elif judge_ == 3:
                    return 0
        self.continue_ = True
        return 0

game = OXgame()
print(game.setstoneO(0))
print(game.BOARD)