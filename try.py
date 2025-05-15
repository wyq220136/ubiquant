import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Value, Lock
import numpy as np

# ----------------------
# 共享经验池 (多进程安全)
# ----------------------
class SharedReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共享内存分配
        self.states = mp.Array('f', buffer_size * state_dim)
        self.actions = mp.Array('f', buffer_size * action_dim)
        self.rewards = mp.Array('f', buffer_size)
        self.next_states = mp.Array('f', buffer_size * state_dim)
        self.dones = mp.Array('i', buffer_size)
        
        self.counter = Value('i', 0)
        self.lock = Lock()

    def add(self, state, action, reward, next_state, done):
        with self.lock:
            idx = self.counter.value % self.buffer_size
            
            # 将数据拷贝到共享内存
            self.states[idx*self.state_dim : (idx+1)*self.state_dim] = state
            self.actions[idx*self.action_dim : (idx+1)*self.action_dim] = action
            self.rewards[idx] = reward
            self.next_states[idx*self.state_dim : (idx+1)*self.state_dim] = next_state
            self.dones[idx] = int(done)
            
            self.counter.value += 1

    def sample(self, batch_size):
        with self.lock:
            current_size = min(self.counter.value, self.buffer_size)
            indices = np.random.choice(current_size, batch_size)
            
            # 从共享内存读取数据
            states = np.array(self.states[:current_size*self.state_dim]).reshape(-1, self.state_dim)[indices]
            actions = np.array(self.actions[:current_size*self.action_dim]).reshape(-1, self.action_dim)[indices]
            rewards = np.array(self.rewards[:current_size])[indices]
            next_states = np.array(self.next_states[:current_size*self.state_dim]).reshape(-1, self.state_dim)[indices]
            dones = np.array(self.dones[:current_size])[indices]
            
            return states, actions, rewards, next_states, dones

# ----------------------
# 训练进程函数
# ----------------------
def train_process(shared_buffer, param_queue, stop_flag, 
                 state_dim, action_dim, device, train_config):
    # 初始化训练用Agent
    trainer = PokerSACAgent(
        memo_capacity=train_config["memory_size"],
        alpha=train_config["alpha"],
        beta=train_config["beta"],
        gamma=train_config["gamma"],
        tau=train_config["tau"],
        batch_size=train_config["batch_size"],
        device=device,
        state_dim=state_dim
    )
    
    # 主训练循环
    while not stop_flag.value:
        # 从共享缓冲区采样
        states, actions, rewards, next_states, dones = shared_buffer.sample(
            train_config["batch_size"]
        )
        
        # 转换为Tensor
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # 执行单步训练
        trainer.train_step(states, actions, rewards, next_states, dones)
        
        # 定期发送参数更新到主进程
        if trainer.steps % train_config["sync_interval"] == 0:
            actor_state = trainer.actor.state_dict()
            critic1_state = trainer.critic1.state_dict()
            critic2_state = trainer.critic2.state_dict()
            param_queue.put(("actor", actor_state))
            param_queue.put(("critic1", critic1_state))
            param_queue.put(("critic2", critic2_state))
            
# ----------------------
# 主进程中的交互Agent
# ----------------------
class InteractiveAgent:
    def __init__(self, state_dim, action_dim, device):
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.device = device
        
    def update_params(self, state_dict):
        self.actor.load_state_dict(state_dict)
        
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_tensor)
        action_id = probs.argmax().item()
        return action_config[action_id]
    
