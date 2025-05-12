import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

action_config = ["giveup", "allin", "check", "callbet", "raisebet"]

class OpponentAnalysis:
    def __init__(self, oppo_id, filter_window=10, gamma=0.3):
        self.id = oppo_id
        self.action_history = [0]*filter_window
        self.agression = 1
        self.idx = 0
        self.gamma = gamma
        self.window_size = filter_window
        
    # 对手这一轮动作录入，0-过牌或弃牌，1-跟注，2-加注
    def calculate_new_agression(self, oppo_action):
        self.action_history[self.idx] = oppo_action
        # td更新
        self.agression = self.agression + self.gamma*(sum(self.action_history)/self.window_size - self.agression)
        

class PokerStateEncoder:
    def __init__(self, identify, player_seat_id=3):
        self.player_seat_id = player_seat_id
        self.max_chips = 2000  # 归一化参考值
        self.max_public_cards = 5  # 最多5张公共牌
        
        # 玩家身份，庄家0，小盲1，大盲2，其他玩家3
        self.identify = identify
        
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
            "SHOWDOWN": 4
        }

    def _parse_card(self, card):
        if isinstance(card, int):  # 处理数值型表示
            rank = card % 13
            suit = card // 13
            return [rank, suit]
        elif isinstance(card, str):
            return [self.rank_map[card[0]], self.suit_map[card[1]]]
        return [0, 0]
    
    def update_identify(self, identify):
        self.identify = identify
    
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
            

    # data从round_info的report函数中获取
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

        # 改到这里，2025.5.12 下午3点
        
        
        
        
        # 筹码特征
        player_chips = next((s["hand_chips"] for s in data["basic_info"]["seat_info"] 
                          if s["seatid"] == self.player_seat_id), 0)
        # opponent_seat = 4 if self.player_seat_id == 3 else 3
        opponent_chips = next((s["hand_chips"] for s in data["basic_info"]["seat_info"] 
                             if s["seatid"] != self.player_seat_id), 0)
        
        # 计算底池
        pot = sum([r.get("bet",0) for r in data["dynamic_info"]["round_history"]])
        
        # 归一化处理
        chip_features = [
            player_chips / self.max_chips,
            opponent_chips / self.max_chips,
            pot / self.max_chips
        ]
        features.extend(chip_features)

        # 4. 位置特征 -------------------------------------------------
        position = [0, 0, 0]  # [庄家, 小盲, 大盲]
        dealer_id = data["basic_info"]["dealer_info"]["seatid"]
        if self.player_seat_id == dealer_id:
            position[0] = 1
        if self.player_seat_id == data["basic_info"]["blind"]["small_blind"]["seatid"]:
            position[1] = 1
        if self.player_seat_id == data["basic_info"]["blind"]["big_blind"]["seatid"]:
            position[2] = 1
        features.extend(position)

        # 5. 阶段特征 -------------------------------------------------
        stage_onehot = [0]*5
        stage_onehot[self.stage_map[current_stage]] = 1
        features.extend(stage_onehot)

        # 6. 对手行为特征 ---------------------------------------------
        opponent_actions = []
        for r in data["dynamic_info"]["round_history"]:
            if r["player"] != self.player_seat_id:
                opponent_actions.append(r.get("type", 0))
        
        # 统计动作类型：0-弃牌，1-跟注，2-加注
        action_counts = [0, 0, 0]
        for a in opponent_actions[-10:]:  # 只看最近10个动作
            if 0 <= a <= 2:
                action_counts[a] += 1
        total = len(opponent_actions) or 1
        aggression = action_counts[2] / total  # 加注频率
        features.append(aggression)

        # 转换为numpy数组并确保维度
        state_vector = np.array(features, dtype=np.float32)
        
        # 最终维度验证
        expected_dim = 2*2 + 5*2 + 3 + 3 + 5 + 1
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
        logits = self.net(torch.FloatTensor(state))
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
        x = F.relu(self.fc1(torch.FloatTensor(state)))
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
        
        critic1_loss.backward()
        critic2_loss.backward()
        
        self.critic_1_optimizer.step()
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
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()
        
        self.alpha_optim.zero_grad()
        # alpha_loss.backward()
        self.alpha_optim.step()
        
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
       
    def memorize(self, action, state, nextstate, reward, done):
        self.memory.add_memory(state, action, nextstate, reward, done)
    
        