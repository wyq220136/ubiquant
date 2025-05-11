class RoundInfoManager:
    def __init__(self, chips):
        self.stage = "PREFLOP"
        self.action_history = []
        self.action_tmp = []
        self.bet_history = []
        self.bet_tmp = []
        
        self.hand_chips = chips
        self.last_cards_num = 0
        
    def update_stage(self, cards_num):
        match cards_num:
            case 0:
                self.stage = "PREFLOP"
            case 3:
                self.stage = "FLOP"
            case 4:
                self.stage = "TURN"
            case 5:
                self.stage = "RIVER"
        self.last_cards_num = cards_num
        self.action_history.append(self.action_tmp)
        self.bet_history.append(self.bet_tmp)
        self.bet_tmp = []
        self.action_tmp = []
        
    def update_round(self, chips):
        self.hand_chips = chips
        self.action_history = []
        self.bet_history = []
        
    def report_history(self):
        if len(self.action_history) >= 2:
            return [self.action_history[-1], self.action_history[-2]], [self.bet_history[-1], self.bet_history[-2]], self.stage
        else:
            return [self.action_history[-1]], [self.bet_history[-1]], self.stage


    def get_extern_info(self, cards_num, act, bet):
        flag = False
        isdone = False
        if cards_num != self.last_cards_num:
            if self.stage == "RIVER":
                isdone = True
            self.update_stage(cards_num)
            flag = True
        self.action_tmp.append(act)
        self.bet_tmp.append(bet)
        return flag, isdone
        
    # 如何判断是跟注还是加注呢
    