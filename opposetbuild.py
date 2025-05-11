import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

action_config = ["giveup", "allin", "check", "callbet", "raisebet"]

class MCCFR_PokerPlayer:
    def __init__(self, action_space):
        self.action_space = action_space
        self.regret = defaultdict(lambda: np.zeros(len(action_space)))
        self.strategy = defaultdict(lambda: np.ones(len(action_space))/len(action_space))
        self.cumulative_strategy = defaultdict(lambda: np.zeros(len(action_space)))

    def update_strategy(self, infoset):
        regret_pos = np.maximum(self.regret[infoset], 0)
        sum_regret = np.sum(regret_pos)
        self.strategy[infoset] = regret_pos / sum_regret if sum_regret > 0 else np.ones_like(regret_pos)/len(regret_pos)

    def train(self, env, sac_agent, iterations=1000):
        for _ in range(iterations):
            self._cfr(env, sac_agent, env.reset(), 1.0, 1.0)

    def _cfr(self, env, sac_agent, state, reach_prob, opp_reach_prob):
        if env.is_terminal(state):
            # 获取u(h)
            return env.get_utility(state)


        infoset = self._get_infoset(state)
        self.update_strategy(infoset)
        current_player = state['current_player']

        if current_player != 0:  # SAC玩家的回合
            action_probs = sac_agent.get_policy(state)  # 需要SAC提供策略接口
            action_idx = np.random.choice(len(self.action_space), p=action_probs)
            next_state = env.step(state, self.action_space[action_idx])
            return self._cfr(env, sac_agent, next_state, reach_prob, opp_reach_prob * action_probs[action_idx])

        # CFR玩家的回合
        node_util = 0
        action_utils = np.zeros(len(self.action_space))
        for a in range(len(self.action_space)):
            action_prob = self.strategy[infoset][a]
            new_reach = reach_prob * action_prob
            next_state = env.step(state, self.action_space[a])
            action_utils[a] = -self._cfr(env, sac_agent, next_state, new_reach, opp_reach_prob)
            node_util += action_prob * action_utils[a]

        for a in range(len(self.action_space)):
            regret = action_utils[a] - node_util
            self.regret[infoset][a] += opp_reach_prob * regret
            self.cumulative_strategy[infoset][a] += reach_prob * self.strategy[infoset][a]
        
        return node_util

    def _get_infoset(self, state):
        return f"{state['public_cards']}_{'_'.join(state['action_history'])}"
    
    
    
    

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
    def __init__(self, state_dim, memo_capacity, alpha, beta, gamma, tau, batch_size, device='cuda'):
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
    
        