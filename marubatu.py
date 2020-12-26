import random

class OXgame:

    def __init__(self):
        self.BOARD = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.continue_ = True
        self.trouble_list = []

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
        if(self.BOARD[i] == 0):
            self.BOARD[i] = 1
            self.trouble_list.append(i)
        else:
            self.trouble_list.append(i)
            return False
            
        
    def setstoneX(self, i):
        if(self.BOARD[i] == 0):
            self.BOARD[i] = 2
            self.trouble_list.append(i)
        else:
            self.trouble_list.append(i)
            return False

    def judge(self, BOARD):
        for i in [0, 3, 6]:
            if (BOARD[i] == 1) and (BOARD[i+1] == 1) and (BOARD[i+2] == 1):
                return 1
            elif (BOARD[i] == 2) and (BOARD[i+1] == 2) and (BOARD[i+2] == 2):
                return 2
        for i in [0, 1, 2]:
            if (BOARD[i] == 1) and (BOARD[i+3] == 1) and (BOARD[i+6] == 1):
                return 1
            elif (BOARD[i] == 2) and (BOARD[i+3] == 2) and (BOARD[i+6] == 2):
                return 2
        if (BOARD[0] == 1) and (BOARD[4] == 1) and (BOARD[8] == 1):
            return 1
        elif (BOARD[0] == 2) and (BOARD[4] == 2) and (BOARD[8] == 2):
            return 2
        if (BOARD[2] == 1) and (BOARD[4] == 1) and (BOARD[6] == 1):
            return 1
        elif (BOARD[2] == 2) and (BOARD[4] == 2) and (BOARD[6] == 2):
            return 2

        judge = True
        for i in range(len(BOARD)):
            if BOARD[i] == 0:
                judge = False
        if judge:
            return 3

        return 0
    
    def cpu(self):
        select_list = []
        return_ = None
        BOARD_copy = self.BOARD.copy()
        for i in range(len(self.BOARD)):
            if self.BOARD[i] == 0:
                select_list.append(i)
        return_ = select_list[random.randrange(len(select_list))]
        if random.randint(0, 100) > 70:
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
    
    def cpu2(self):
        select_list = []
        return_ = None
        BOARD_copy = self.BOARD.copy()
        for i in range(len(self.BOARD)):
            if self.BOARD[i] == 0:
                select_list.append(i)
        return_ = select_list[random.randrange(len(select_list))]
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
    
    def game_step(self, action):
        self.continue_ = False
        if self.setstoneO(action) == False:
                return -1
        else:
            judge_ = self.judge(self.BOARD)
            if judge_ == 1:
                return 1
            elif judge_ == 3:
                return 0
            if self.setstoneX(self.cpu()) == True:
                return 1
            else:
                judge_ = self.judge(self.BOARD)
                if judge_ == 2:
                    return -1
                elif judge_ == 3:
                    return 0
        self.continue_ = True
        return 0