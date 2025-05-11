from typing import Dict, List, Optional, Union
from detect import player_info
from evaluation_new import evaluation_net
import torch

legal_action = ["giveup", "allin", "check", "callbet", "raisebet"]

class cfrp:
    def __init__(self, player:player_info, player_name, threshold:float):
        self.initial_regret = {action:0 for action in legal_action}
        self.voa = dict()
        # 返回player的hand_chips为总收益
        self.hero = player
        self.hero_name = player_name
        self.explored = {action: False for action in legal_action}
        
        # 动作选择边界，大于此后悔值选择动作
        self.threshold = threshold
        # 大语言模型评估器
        self.evaluater = evaluation_net()
        
        self.evaluater.load_state_dict(torch.load("best_model.pth"))
        self.evaluater.eval()
        
        self.agent = Agent()
        
    
    def calculate_strategy(self, this_info_sets_regret: Dict[str, float]) -> Dict[str, float]:
        # TODO: Could we instanciate a state object from an info set?
        actions = this_info_sets_regret.keys()
        regret_sum = sum([max(regret, 0) for regret in this_info_sets_regret.values()])
        if regret_sum > 0:
            strategy: Dict[str, float] = {
                action: max(this_info_sets_regret[action], 0) / regret_sum
                for action in actions
            }
        else:
            default_probability = 1 / len(actions)
            strategy: Dict[str, float] = {action: default_probability for action in actions}
        return strategy
    
    
    # state, 网络推理用结构体，两者结构一致
    def generate_regret(self, state):
        dict_info = {}
        for i in legal_action:
            value = self.evaluater(state, i)
            dict_info[i] = value
        if dict_info == {}:
            return self.initial_regret
        else:
            return dict_info
        
    
    def run(self, state, legal_actions:list):
        vo = 0
        if not self.hero.is_active:
            return self.hero.hand_chips
        if state.isdone:
            return self.hero.hand_chips
        # 初始用网络生成结果
        this_info_sets_regret = self.generate_regret(state)
        sigma = self.calculate_strategy(this_info_sets_regret)
        for action in legal_actions:
            if this_info_sets_regret[action] > self.threshold:
                self.voa[action] = self.evaluater(state, action)
                self.explored[action] = True
                vo += sigma[action] * self.voa[action]
        this_info_sets_regret = self.generate_regret(state)
        for action in legal_actions:
            if self.explored[action]:
                this_info_sets_regret[action] += self.voa[action] - vo
        self.agent.regret[state] = this_info_sets_regret
        
        return vo
    
    
class Agent:
    def __init__(self):
        self.regret = {}
