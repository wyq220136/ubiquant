import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
# from collections import defaultdict

action_config = ["giveup", "allin", "check", "callbet", "raisebet"]

class OpponentAnalysis:
    def __init__(self, oppo_id, filter_window=10, gamma=0.3):
        self.id = oppo_id
        self.action_history = [0]*filter_window
        self.agression = 1
        self.idx = 0
        self.gamma = gamma
        self.window_size = filter_window
        self.hand_chips = 2000
        
        self.round_agression = []
        
    # 对手这一轮动作录入，0-过牌或弃牌，1-跟注，2-加注
    def calculate_new_agression(self, oppo_action):
        self.action_history[self.idx] = oppo_action
        self.round_agression.append(oppo_action)
        # td更新
        self.agression = self.agression + self.gamma*(sum(self.action_history)/self.window_size - self.agression)
        
    def clean_round_record(self):
        self.round_agression = []
        
    # 计算对手当这一局的侵略性
    def calculate_agression_now(self):
        return sum(self.round_agression)/len(self.round_agression)
        
# 只对外开放encode函数用于处理局内信息，其余用于维护
class PokerStateEncoder:
    def __init__(self, player_seat_id=3, number=2):
        self.player_seat_id = player_seat_id
        self.max_chips = 2000  # 归一化参考值
        
        self.player_number = number
        
        # 静态信息是否初始化完成
        self.static_is_load = False
        
        # 映射表初始化
        self.rank_map = {'2':0, '3':1, '4':2, '5':3, '6':4,
                        '7':5, '8':6, '9':7, 'T':8, 'J':9,
                        'Q':10, 'K':11, 'A':12}
        self.suit_map = {'c':0, 'd':1, 'h':2, 's':3}
        self.stage_map = {
            "PREFLOP": 0,
            "FLOP": 1,
            "TURN": 2,
            "RIVER": 3,
        }
        
        self.nickDict = {}
        
        self.info_storage = info_storage()
        
        self.reward_wrapper = rewardwrapper()


    def load_all_nick(self, data):
        if not self.static_is_load:
            if data["blind"]["big_blind"]["seatid"] == self.player_seat_id:
                self.identify = 2
            elif data["blind"]["small_blind"]["seatid"] == self.player_seat_id:
                self.identify = 1
            elif data["dealer_info"]["seatid"] == self.player_seat_id:
                self.identify = 0
            else:
                self.identify = 3
            for id in data["seat_info"]:
                if id["seatid"] != self.player_seat_id:
                    self.nickDict[id["seatid"]] = OpponentAnalysis(id)
            self.round_action = [0]*len(self.nickDict)
            # print("11111", self.nickDict.keys())
            self.static_is_load = True
        
        
    def _parse_card(self, card):
        if isinstance(card, int):  # 处理数值型表示
            rank = card % 13
            suit = card // 13
            return [rank, suit]
        elif isinstance(card, str):
            return [self.rank_map[card[0]], self.suit_map[card[1]]]
        return [0, 0]
        
    
    # 用于公共牌组和私人手牌编码
    def encode_cards(self, cards:list)->list:
        cards_modified = []
        # 只有手牌才能出现空列表情况
        if cards == []:
            return [[0, 0]]*2
        for card in cards:
            if card != -1:
                card_tmp = self._parse_card(card)
                cards_modified.append(card_tmp)
            else:
                cards_modified.append([0, 0])
        return cards_modified
    
    
    def update_oppoaction(self, oppo_action, id):
        self.nickDict[id].calculate_new_agression(oppo_action)
        # self.round_action[id] = oppo_action
        
    def update_chips(self, oppo_chips, id):
        # print(self.nickDict.keys())
        self.nickDict[id].hand_chips = oppo_chips
        
        
    def encode_agression(self):
        agression_all = []
        agression_now = []
        for i in self.nickDict.values():
            agression_all.append(i.agression)
            agression_now.append(i.calculate_agression_now())
        return agression_all, agression_now

    # data从round_info的report函数中获取,只有轮到我才会编码
    def encode(self, data):
        features = []

        # 手牌特征
        player_hand = data["hand_cards"]
        hand_feature = self.encode_cards(player_hand)
        features.extend(hand_feature)

        # 公共牌特征
        table_cards = data["table_cards"]
        public_feature = self.encode_cards(table_cards)
        features.extend(public_feature)
        
        # 筹码特征
        player_chips = next(data["player_chips"], 0)
        
        # 全部对手筹码放入列表
        opponent_chips = []
        opponent_chips.append(s.hand_chips for s in self.nickDict.values())
        
        # 计算底池
        pot = data["all_bet"]
        
        # 归一化处理
        chip_features = [
            player_chips / self.max_chips,
            pot / self.max_chips
        ]
        chip_features.extend(opponent_chips)
        features.extend(chip_features)

        # 位置特征, 采用独热编码
        position = [0, 0, 0]  # [庄家, 小盲, 大盲]
        match self.identify:
            case 0:
                position[0] = 1
            case 1:
                position[1] = 1
            case 2:
                position[2] = 1
        features.extend(position)
        
        # 阶段特征
        stage_onehot = [0]*4
        stage_onehot[self.stage_map[data["current_stage"]]] = 1
        features.extend(stage_onehot)
        
        # 对手特征
        aggress_all, aggress_now = self.encode_agression()
        features.extend(aggress_all)
        features.extend(aggress_now)

        # 转换为numpy数组并确保维度
        state_vector = np.array(features, dtype=np.float32)
        
        # 最终维度验证
        expected_dim = 2*2 + 5*2 + self.player_number + 1 + 3 + 4 + 2*(self.player_number-1)
        assert len(state_vector) == expected_dim, \
            f"维度错误: 预期{expected_dim}, 实际{len(state_vector)}"
        
        return state_vector
    

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=256, hidden_dim2=256, lr=0.0001):
        super(ActorNetwork, self).__init__()
        # 修改网络结构适配离散动作
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, action_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        logits = self.net(state)
        return F.softmax(logits, dim=-1)  # 输出动作概率分布

    def sample_action(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), F.one_hot(action, num_classes=len(action_config)).float()
        
        
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=256, hidden_dim2=256, lr=0.001):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.q = nn.Linear(hidden_dim2, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        
        return q
  
    
class ReplayMemory:
    def __init__(self, memory_size, state_dim, action_dim):
        self.memo_size = memory_size
        self.state_memo = np.zeros((memory_size, state_dim))
        self.nextstate_memo = np.zeros((memory_size, state_dim))
        self.action_memo = np.zeros((memory_size, action_dim))
        self.reward_memo = np.zeros(memory_size)
        self.done_memo = np.zeros(memory_size)
        self.counter = 0
        
    def add_memory(self, state, action, reward, next_state, done):
        idx = self.counter % self.memo_size
        self.state_memo[idx] = state
        self.nextstate_memo[idx] = next_state
        self.action_memo[idx] = action
        self.reward_memo[idx] = reward
        self.done_memo[idx] = done
        
        self.counter += 1
    
    def sample_memory(self, batch_size):
        current_batch_size = min(self.memo_size, self.counter)
        if current_batch_size < batch_size:
            batch = np.random.choice(current_batch_size, batch_size, replace=True)
        else:
            batch = np.random.choice(current_batch_size, batch_size, replace=False)
        batch_state = self.state_memo[batch]
        batch_nextstate = self.nextstate_memo[batch]
        batch_action = self.action_memo[batch]
        batch_reward = self.reward_memo[batch]
        batch_done = self.done_memo[batch]
        
        return batch_state, batch_nextstate, batch_action, batch_reward, batch_done
        


class PokerSACAgent:
    def __init__(self, memo_capacity, alpha, beta, gamma, tau, batch_size, device='cuda', state_dim=26):
        self.action_dim = 5
        self.memory = ReplayMemory(memo_capacity, state_dim, self.action_dim)
        self.actor = ActorNetwork(state_dim, self.action_dim).to(device)
        self.critic1 = CriticNetwork(state_dim, self.action_dim).to(device)
        self.critic2 = CriticNetwork(state_dim, self.action_dim).to(device)
        
        self.target_critic1 = CriticNetwork(state_dim, self.action_dim).to(device)
        self.target_critic2 = CriticNetwork(state_dim, self.action_dim).to(device)
        
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = device

        # 温度参数优化器
        self.target_entropy = -np.log(1/self.action_dim)  # 离散动作目标熵
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=beta)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=0.0001)
        self.critic_2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=0.0001)
        
    def soft_update(self, net, target_net):
        # 遍历预测网络和目标网络的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data*(1-self.tau) + param.data*self.tau)

    def select_action(self, state):
        # 根据actor选择最佳动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action_id = action_dist.sample()
        action_onehot = F.one_hot(action_id, num_classes=self.action_dim).squeeze(0).cpu().numpy()
        return action_config[action_id.item()], action_onehot

    def train(self):
        states, next_states, actions, rewards, dones = self.memory.sample_memory(self.batch_size)
        # 转换数据到设备
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions.argmax(1)).to(self.device)  # 转换one-hot到索引
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones.astype(np.float32)).to(self.device)
        
        self.alpha = self.log_alpha.exp().detach()  # 从log_alpha计算实际alpha值
        
        # 修改Critic更新逻辑
        with torch.no_grad():
            next_probs = self.actor(next_states)
            next_log_probs = torch.log(next_probs + 1e-8)
            
            entropy = -torch.sum(next_probs * next_log_probs, dim=-1, keepdim=True)
            
            # 计算目标Q值
            q1_next = self.target_critic1(next_states)
            q2_next = self.target_critic2(next_states)
            q_next = torch.sum(next_probs*torch.min(q1_next, q2_next), dim=-1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * (q_next - self.alpha * entropy)
        
        # 更新Critic
        current_q1 = self.critic1(states).gather(1, actions.unsqueeze(1))
        current_q2 = self.critic2(states).gather(1, actions.unsqueeze(1))
        critic1_loss = torch.mean(F.mse_loss(current_q1, target_q))
        critic2_loss = torch.mean(F.mse_loss(current_q2, target_q))
        
        
        self.critic_1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic_1_optimizer.step()
        
        
        self.critic_2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic_2_optimizer.step()
                
        # 更新Actor
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q1 = self.critic1(states)
            q2 = self.critic2(states)
            q = torch.min(q1, q2)
        actor_loss = (probs * (self.alpha * log_probs - q)).sum(dim=1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新温度参数
        entropy = -torch.sum(probs * log_probs, dim=-1)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
       
    def memorize(self, state, action, nextstate, reward, done):
        self.memory.add_memory(state, action, nextstate, reward, done)
    

# 读取采取行动后奖励的方法，记录本回合决策到下回合决策之间发生的事情，合成动作的奖励
class rewardwrapper:
    def __init__(self):
        self.giveup_num = 0
        self.delta_bet = 0
        self.is_act = False
        self.win_reward = 0
        
        
    def get_reward(self, instance):
        if self.is_act:
            if instance["Type"] == 5:
                self.giveup_num += 1
            
            else:
                self.delta_bet += instance["Bet"]
        
        
    def refresh(self):
        self.giveup_num = 0
        self.delta_bet = 0
        self.win_reward = 0
        self.is_act = True
    
    
    def calculate_reward(self):
        self.is_act = False
        return 0.01*self.delta_bet+0.8*self.giveup_num + self.win_reward
    
    
    # win_bet有正有负
    def is_win(self, res, win_bet):
        if res:
            self.win_reward = win_bet
        
        else:
            self.win_reward = win_bet
    
    
class info_storage:
    def __init__(self):
        self.state = None
    
    def update_state(self, state):
        self.state = state
        
    def update_action(self, action):
        self.action = action
    
    def update_reward(self, reward):
        self.reward = reward
        
    def isdone(self):
        self.dones = 1
        
    def update_nextstate(self, nextstate):
        self.nextstate = nextstate
        
    def report(self):
        return self.state, self.nextstate, self.action, self.reward, self.dones
      
    def refresh(self):
        self.dones = 0

# class judge_winner:
#     def __init__(self):
#         self.is_win = False