
import numpy as np
import random
import torch
import math
from torch import nn
from torch import optim
import torch.nn.functional as F
from DRL_environment import Environment
import matplotlib.pyplot as plt
#plt.ion()
from mpl_toolkits.axes_grid1 import host_subplot
from torch.autograd import Variable
import time
#from layers import NoisyLinear
#from NaivePrioritizedBuffer import NaivePrioritizedBuffer
torch.manual_seed(123)
from multiprocessing.managers import BaseManager
import json
from argparse import ArgumentParser
import torch.multiprocessing as mp
import random
random.seed(123)
from layers import NoisyLinear
import numpy as np
from multiprocessing import Manager,Process, Queue
np.random.seed(123)
import sys
sys.setrecursionlimit(100000) #迭代深度设置为十万 
from collections import namedtuple
alpha=0.6
num_atoms = 50
Vmin = -10
Vmax = 10
num_states = 4  # set the number of state and action
num_actions = 2
num_actors = 2

#training

num_frames = 2000
BATCH_SIZE = 250
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

#
MAX_TASKS = 10#任务数
NUM_EPISODES = 10
#  50_888 BATCH_SIZE = 80  200
#BATCH_SIZE = 250
CAPACITY = 2000# 记忆库容量
EPS_START = 1.0#贪婪参数初始值
EPS_END = 0.1#贪婪参数最小值
EPS_DECAY = 10000#贪婪参数变化次数
TARGET_REPLACE_ITER = 5#target net更新次数
steps_done = 0
eval_profit_list = []
avg_profit_list=[]
eval_tasks_list=[]
avg_tasks_list=[]
eval_ap_list=[]
avg_ap_list=[]


def plot_profit(profit,avg):
    host = host_subplot(111)  
    plt.subplots_adjust(right=0.8) 

    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total profit")
       
    # plot curves
    p1, = host.plot(range(len(profit)), profit, label="Total Profit")
    p2, = host.plot(range(len(avg)), avg, label="Running Average Total Profit")
    host.legend(loc=1)
 
    # set color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
   
    host.set_xlim([0, NUM_EPISODES])
    host.set_ylim([0,500])

 
    plt.draw()
    plt.show()

def plot_tasks(tasks,avg1):
    host = host_subplot(111)  
    plt.subplots_adjust(right=0.8)  

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Total accepted tasks")
 

    # plot curves
    p1, = host.plot(range(len(tasks)), tasks, label="Total Accepted Tasks")
    p2, = host.plot(range(len(avg1)), avg1, label="Running Average Total Accepted Tasks")
 

    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    # set the range of x axis of host and y axis of par1
    host.set_xlim([0,NUM_EPISODES])
    host.set_ylim([0,MAX_TASKS])
 
    plt.draw()
    plt.show()
    
def plot_ap(ap,avg2):
    host = host_subplot(111) 
    plt.subplots_adjust(right=0.8)  

 
    # set labels
    host.set_xlabel("Training episdoes")
    host.set_ylabel("Average Profit")
 
    # plot curves
    p1, = host.plot(range(len(ap)), ap, label="Average Profit")
    p2, = host.plot(range(len(avg2)), avg2, label="Running Average Profit")
 
    # set location 
    host.legend(loc=1)
 
    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    host.axis["right"].label.set_color(p2.get_color())
    host.set_xlim([0,NUM_EPISODES])
    host.set_ylim([0,9])
 
    plt.draw()
    plt.show()

class RainbowDQN(nn.Module):
    def __init__(self, num_states, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowDQN, self).__init__()
        
        self.num_states   = num_states
        self.num_actions  = num_actions
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax
        
        self.linear1 = nn.Linear(num_states, 32)#输入出
        self.linear2 = nn.Linear(32, 64)#输入层
        
        self.noisy_value1 = NoisyLinear(64, 64,)# 噪声层
        self.noisy_value2 = NoisyLinear(64, self.num_atoms)
        
        self.noisy_advantage1 = NoisyLinear(64, 64)#优势函数
        self.noisy_advantage2 = NoisyLinear(64, self.num_atoms * self.num_actions)

        #self.current_model = RainbowDQN(num_states, num_actions, num_atoms, Vmin, Vmax)
        #self.target_model  = RainbowDQN(num_states, num_actions, num_atoms, Vmin, Vmax)

        #self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.00068)
    
    # 各层对应的激活函数
    def forward(self, x):
        BATCH_SIZE = x.size(0)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        
        value     = value.view(BATCH_SIZE, 1, self.num_atoms)
        advantage = advantage.view(BATCH_SIZE, self.num_actions, self.num_atoms)
        #正儿八经的的Q值，就是由value和advantage确定的
        x = value + advantage - advantage.mean(1, keepdim=True)#x.mean()批量归一化
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
        
        return x
        
    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    #原代码，epslion= 0.5*(1/(episode+1))这个没看懂是啥
    #def decide_action(self,state,episode):
        #epslion = 0.5*(1/(episode+1))
        #if epslion<= np.random.uniform(0,1):
            #self.0.eval()
            #with torch.no_grad():
                #action=self.model(state).max(1)[1].view(1,1)
        #else:
            #action = torch.LongTensor([[random.randrange(self.num_actions)]])
        #return action

    def act(self,state):
        global steps_done
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        sample = random.random()## 产生 0 到 1 之间的随机浮点数
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1.*steps_done / EPS_DECAY)#最小到0.427
        steps_done += 1
        if sample > eps_threshold:#判断是随即动作还是最优动作
        #sample是(0，1)，eps_threshold越来越小，一开始是选择最优策略（开发）
            #self.model.eval()
            with torch.no_grad():#torch.no_grad()一般用于神经网络的推理阶段, 表示张量的计算过程中无需计算梯度
                action = dist.sum(2).max(1)[1].numpy()[0]#使用最优动作
        else:
            #到后期会越来越趋向于（探索），u而就是随机选择一个动作。
            action = torch.FloatTensor([[random.randrange(self.num_actions)]])#random.randrange（N）在0-N之间随机生成一个数，N是动作空间数
        return action



# current_model = RainbowDQN(num_states, num_actions, num_atoms, Vmin, Vmax)
# target_model  = RainbowDQN(num_states, num_actions, num_atoms, Vmin, Vmax)

# #optimizer = optim.Adam(current_model.parameters(), lr=0.00001)
# optimizer = torch.optim.RMSprop(current_model.parameters(), lr=0.00025 / 4, weight_decay=0.95, eps=1.5e-7)

Transition1 = namedtuple(
    'Transition1', ('state', 'action', 'next_state', 'reward'))

#Transition保存的是每一步的状态、动作、奖励、
# 下一步的折扣因子以及状态-动作价值函数q
Transition = namedtuple('Transition', ['S', 'A', 'R', 'Gamma', 'q'])
N_Step_Transition = namedtuple('N_Step_Transition', ['S_t', 'A_t', 'R_ttpB', 'Gamma_ttpB', 'qS_t', 'S_tpn', 'qS_tpn', 'key'])
#N_Step_Transition在Transition的基础上，加入了n-step回报的相关信息
# 当前状态（S_t）、当前动作（A_t）、从当前状态开始的n-step回报（R_ttpB）、
# 从当前状态开始的n步折扣系数（Gamma_ttpB）、当前状态的q值（qS_t）、
# n步后的状态（S_tpn）、n步后状态的q值（qS_tpn,评估动作的价值）以及一个用于区分不同n步序列的key。

#构建全局缓冲池
class ReplayMemory(object):
    def __init__(self, soft_capacity):
        self.soft_capacity = soft_capacity
        self.memory = list()
        self.mem_idx = 0
        self.counter = 0
        #self.alpha = params['priority_exponent']
        self.alpha = 0.6
        self.priorities = dict()
        self.sample_probabilities = dict()
        #self.global_buffer= list()

    def update_sample_probabilities(self):
        # 更新从重放经验池中抽取经验的概率
        
        priorities = self.priorities
        #计算每个经验的概率。每个经验的优先级p，概率为p**alpha除以所有经验优先级的和。
        prob = [p**self.alpha/ sum(priorities.values())  for p in priorities.values()]
        #将经验概率进行归一化
        prob /= sum(prob)
        #将更新后的概率赋值给sample_probabilities。
        # priorities是一个字典，它的键是经验的ID，值是该经验的概率。
        self.sample_probabilities.update({k:v for k in list(priorities) for v in prob})
        # 计算所有经验的概率的和
        sum_of_prob = sum(self.sample_probabilities.values())
        #将所有经验的概率进行归一化，使它们的和为1。
        # 这里将字典中的每个值除以sum_of_prob来实现归一化。
        for k in self.sample_probabilities.keys():
            self.sample_probabilities[k] /= sum_of_prob

    def set_priorities(self, new_priorities):
        #使用唯一标识经验的key键更新经验的优先级。
        # 如果key键不存在，则添加它。如果key键已经存在，则更新/覆盖优先级值。
        #每当更改/添加优先级时，样本概率也会更新。
        #将参数new_priorities更新到对象的priorities属性中
        self.priorities.update(new_priorities)
        #更新采样概率，以反映新的优先级权重的变化
        self.update_sample_probabilities()


    def add(self, priorities, xp_batch):
        #向重放经验池添加批量经验和​​优先级
        #:param priorities: xp_batch中经验的优先级
        #:param xp_batch: N_Step_Transitions 类型的体验列表

        # 将新的经验数据添加到经验池
        for xp in xp_batch:
            self.memory.append(xp)
        #[self.memory.append(xp) for xp in xp_batch]
        # 使用 set_priorities 设置新体验的初始优先级，它也负责更新概率
        self.set_priorities(priorities)

    def push(self, state, action, state_next, reward):

        if len(self.memory) < CAPACITY:
            self.memory.append(None)  
        self.memory[self.mem_idx] = Transition1(state, action, state_next, reward)

        self.index = (self.mem_idx + 1) % CAPACITY
    

    def sample(self, sample_size):
        #根据使用采样优先级计算出的采样概率，返回从重放经验池采样的一批经验。
        #:param sample_size: 要从优先缓冲区中采样的批次的大小
        #:return: N_Step_Transition 对象列表
        mem = N_Step_Transition(*zip(*self.memory))
        # 获取priorities字典的键并将其转换为列表
        keys = list(self.priorities.keys())
        # 获取sample_probabilities字典的值并将其转换为列表
        probs = list(self.sample_probabilities.values())
        # 从keys列表中随机抽样sample_size次，每次抽样使用probs作为概率分布
        sampled_keys = [np.random.choice(keys, p=probs) for _ in range(sample_size)]

        #sampled_keys = [np.random.choice(list(self.priorities.keys()), p=list(self.sample_probabilities.values()))
                        #for _ in range(sample_size) ]
        batch_xp = [N_Step_Transition(S, A, R, G, qt, Sn, qn, key) 
                    for k in sampled_keys
                    for S, A, R, G, qt, Sn, qn, key in zip(mem.S_t, mem.A_t, mem.R_ttpB, mem.Gamma_ttpB,
                                                           mem.qS_t, mem.S_tpn, mem.qS_tpn, mem.key) 
                    if key == k]
        return batch_xp

    def remove_to_fit(self):
        #删除超过软容量阈值的重放经验池数据的方法。经验按 FIFO 顺序删除
        #This method由学习器定期调用
        
        if self.size() > self.soft_capacity:
            num_excess_data = self.size() - self.soft_capacity
            # FIFO
            del self.memory[: num_excess_data]

    def size(self):
        return len(self.memory)


    # def get(self, batch_size):
    #     #assert请求的n步atch_size大小不超过缓冲区中可用的数量，如果请求的批量大小超过了可用的数量，抛出一个异常。
    #     assert batch_size <= self.size, "Requested n-step transitions batch size is more than available"
    #     #将local_nstep_buffer中的前batch_size个n步transition取出，保存到一个batch_of_n_step_transitions列表中
    #     batch_of_leaner_transitions = self.global_buffer[: batch_size]
    #     #删除local_nstep_buffer中的前batch_size个n步transition
    #     del self.global_buffer[: batch_size]
    #     #返回batch_of_n_step_transitions列表，其中保存着获取到的n步transition。
    #     return batch_of_leaner_transitions



#局部缓冲池
#实现了环形缓冲区，用于保存每个actor执行的n-step transition，以用于后面的训练
class ExperienceBuffer(object):
    #actor_id是用于区分不同actor的ID
    def __init__(self, capacity, actor_id):
        #一个actor执行的一步transition,存储单步经验的列表，以组成 n 步转换。
        self.local_1step_buffer = list() 
        #2是n-step大小
        #self.local_1step_buffer = [Transition(None, None, None, None, None) for _ in range(2)]
        #一个actor执行的n-step transition（即多个local_1step_buffer）
        self.local_nstep_buffer= list()  
        #idx是当前存储到local_1step_buffer的位置索引
        self.idx = -1
        #local_1step_buffer的最大容量
        self.capacity = capacity
        self.gamma = 0.99
        self.id = actor_id
        #用于区分不同n步序列
        self.n_step_seq_num = 0  # Used to compose the unique key per per-actor and per n-step transition stored


    #每次环境与多Actor的交互，都会生成一个 1-step 的 transition（状态转移），
    # 该函数将这些 1-step 的 transition 保存在一个本地的缓存 self.local_1step_buffer 中，
    # 然后将其转化为 n-step 的 transition（n步状态转移），
    # 并存储到另一个本地缓存 self.local_nstep_buffer 中。
                                                                                                                                                
    def update_buffer(self):
        #self.local_buffer = Transition(*zip(*xp_batch))
        #循环遍历当前缓冲区中的所有单步转换 transition。
        for i in range(self.B - 1):
            #当前动作的即时奖励
            R = self.local_1step_buffer[i].R
            Gamma = 1
            #对于每个单步转换，计算从该转换开始的 n 步转换的奖励和折扣。
            for k in range(i + 1, self.B ):
                Gamma *= self.gamma
                #使用这些值来创建一个新的 n 步batch_size，并将其添加到本地 n 步缓冲区中。
                R += Gamma * self.local_1step_buffer[k].R
            self.local_1step_buffer[i] = Transition(self.local_1step_buffer[i].S,
                                                    self.local_1step_buffer[i].A, R, Gamma,
             
                                                    self.local_1step_buffer[i].q)

   #将local_1step_buffer中的单步经验转化为n步经验
    def construct_nstep_transition(self, data):
    
        if self.idx == -1:  #  Episode ended at the very first step in this n-step transition
            return
        #为该n-step transition分配一个唯一的key，第i个actor的第j个n步序列号
        key = str(self.id) + str(self.n_step_seq_num)
        #构造一个n-step transition
        #因为local_1step_buffer[0]是一个包含5个元素的元组，而 data.S，data.q 和 key 代表n-step transition的信息，
        # 这些加上 self.local_1step_buffer[0] 中存储的1-step transition信息，可以构成完整的n步信息
        n_step_transition = N_Step_Transition(*self.local_1step_buffer[0], data.S, data.q, key)
        #n-step序列号++
        self.n_step_seq_num += 1
        #  Put the n_step_transition into 本地缓存 self.local_nstep_buffer 中
        self.local_nstep_buffer.append(n_step_transition)
        #  清空 self.local_1step_buffer，
        self.local_1step_buffer.clear()
        #  Reset 准备接收下一个 n-step 的 transition。
        self.idx = -1

    def add(self, data):
        #如果单步transition缓冲区小于最大容量，idx属性增加1，将新的单步transition添加到缓冲区中。
        if self.idx  + 1 < self.capacity:
            self.idx += 1
            #将一个None添加到local_1step_buffer中，以保持其大小与最大容量一致
            self.local_1step_buffer.append(None)
            #将新的单步transition添加到local_1step_buffer中的idx位置
            self.local_1step_buffer[self.idx] = data
            #调用update_buffer方法，计算累积逐步折扣gamma和部分回报。
            self.update_buffer()  
        else:  
            #  调用construct_nstep_transition方法，,将所有单步transition构建为一个n步transition
            self.construct_nstep_transition(data)


    def get(self, batch_size):
        #assert请求的n步atch_size大小不超过缓冲区中可用的数量，如果请求的批量大小超过了可用的数量，抛出一个异常。
        assert batch_size <= self.size, "Requested n-step transitions batch size is more than available"
        #将local_nstep_buffer中的前batch_size个n步transition取出，保存到一个batch_of_n_step_transitions列表中
        batch_of_n_step_transitions = self.local_nstep_buffer[: batch_size]
        #删除local_nstep_buffer中的前batch_size个n步transition
        del self.local_nstep_buffer[: batch_size]
        #返回batch_of_n_step_transitions列表，其中保存着获取到的n步transition。
        return batch_of_n_step_transitions


    @property
    def B(self):
       #单步缓冲区的前大小
        return len(self.local_1step_buffer)

    @property
    def size(self):
        #当前本地n步经验缓冲区的大小
        return len(self.local_nstep_buffer)



#实例化全局缓冲池
global_replay_queue = ReplayMemory(10000)

class MultiActor(mp.Process):
    def __init__(self, actor_id, shared_state, actor_params):
        super(MultiActor, self).__init__()
        self.actor_id = actor_id  # Used to compose a unique key for the transitions generated by each actor
        
        self.params = actor_params
        #共享状态
        self.shared_state = shared_state

        #self.T = self.params["T"]
        self.Q = RainbowDQN(num_states, num_actions, num_atoms, Vmin, Vmax)
        #获取最新网络参数，Q_state_dict 是 shared_state 字典中的一个键，对应的值是target_model神经网络的权重参数
        self.Q.load_state_dict(shared_state["Q_state_dict"])

        self.env = Environment()

        self.policy = self.epsilon_greedy_Q
        #3是num_steps
        self.local_experience_buffer = ExperienceBuffer(3, self.actor_id)
        
        eps = self.params['epsilon']
        num_actors = self.params['num_actors']
        alpha = 0.6
        self.epsilon = eps**(1 + alpha * self.actor_id / (num_actors-1))
        self.gamma = self.params['gamma']
        self.num_buffered_steps = 0  
        # Used to compose a unique key for the transitions generated by each actor
        

    def epsilon_greedy_Q(self, qS_t):
        if random.random() >= self.epsilon:
            #print("np.max(qS_tpn,1):", np.argmax(qS_t))
            return np.argmax(qS_t)
        else:
            return random.choice(list(range(len(qS_t))))

    def compute_priorities(self, n_step_transitions):
        n_step_transitions = N_Step_Transition(*zip(*n_step_transitions))
        # Convert tuple to numpy array
        rew_t_to_tpB = np.array(n_step_transitions.R_ttpB)#n-step回报
        gamma_t_to_tpB = np.array(n_step_transitions.Gamma_ttpB)#未来折扣因子
        qS_tpn = np.array(n_step_transitions.qS_tpn, dtype=object)#n步之后状态的动作值q函数
        A_t = np.array(n_step_transitions.A_t, dtype=object)#当前动作
        qS_t = np.array(n_step_transitions.qS_t,dtype=object)#当前状态的动作值q函数
        
        maxtpn = []
        for i in range(len(qS_tpn)):
            maxtpn.append(qS_tpn[i].max())
            #print(maxtpn)

        max = np.max(maxtpn)
        # 计算 n-step TD errors
        # print(rew_t_to_tpB)
        # print(gamma_t_to_tpB)
        n_step_td_target =  rew_t_to_tpB + gamma_t_to_tpB * maxtpn

        #print(n_step_td_target)

        nparr = []
        for i in range(A_t.shape[0]) :
            #print(qS_t[i][0][int(A_t[i])])
            nparr.append(qS_t[i][0][int(A_t[i])])
        #print("td_target:", n_step_td_target)
        n_step_td_error = n_step_td_target - nparr
        #print(n_step_td_error)
        #print("td_err:", n_step_td_error)
        priorities = dict()
        #print(n_step_transitions)
        for i in range(len(n_step_transitions.key)):
            priorities[n_step_transitions.key[i]]= n_step_td_error[i]
        #print(priorities)
        print("priorities", priorities)
        return priorities


    def run(self):
        
        for episode in range(NUM_EPISODES):
            self.env.reset()
            observation = self.env.observe()  

            ep_reward = []
            #accepted_tasks=0
            #计算当前状态的动作价值函数。
            state = observation
            state = torch.from_numpy(state).type(
                torch.FloatTensor)  
            state = torch.unsqueeze(state, 0)
            #每个回合的循环
            for step in range(MAX_TASKS-1):
                #选择动作
                action = self.policy(state)
                #执行下一个动作，获得下一个观测值、奖励和完成状态。
                next_observation, reward, done = self.env.update_env(
                    action.item()) 
                #将当前状态、动作、奖励、折扣因子和动作价值函数添加到关系缓冲区中。
                self.local_experience_buffer.add(Transition(observation, action, reward , self.gamma, state))
                
                ep_reward.append(reward)
                next_state = next_observation
                    
                next_state = torch.from_numpy(next_state).type(
                        torch.FloatTensor)  
                next_state = torch.unsqueeze(next_state, 0)
                #print("Actor#", self.actor_id,  "NUM_EPISODES=", NUM_EPISODES, "action=", action, "reward:", reward, "1stp_buf_size:", self.local_experience_buffer.B, end='\r')
                
                #重置
                if done: 
                    next_state = None
                    self.local_experience_buffer.construct_nstep_transition(Transition(observation, action, reward, self.gamma, state))
                    #self.env.reset()
                    #observation = self.env.observe()
                    #print("Actor#:", self.actor_id, "  ep_len:", len(ep_reward), "  ep_reward:", np.sum(ep_reward))
                    #:重置周期奖励列表。
                    ep_reward = []
                    break
                 

                #如果局部经验缓冲区的大小大于或等于经验缓冲区transition大小，则执行以下操作。
                if self.local_experience_buffer.size >= self.params['n_step_transition_batch_size']:
                # 从局部经验缓冲区中获取多步转移transition。
                    n_step_experience_batch = self.local_experience_buffer.get(self.params['n_step_transition_batch_size'])
                
                # 计算经验的优先级。
                    priorities = self.compute_priorities(n_step_experience_batch)

               # 将n步transition和它们的优先级添加到全局缓冲池中
                    global_replay_queue.add(priorities, n_step_experience_batch)
                # for i in range(len(n_step_experience_batch)):
                #     priorities = priorities[n_step_experience_batch[i].key]
                #     self.global_replay_queue_queue.add(priorities, n_step_experience_batch)

                # self.global_replay_queue_queue.add([priorities, n_step_experience_batch])
                state = next_state

            #果步数是Q网络同步频率的倍数，则执行以下操作。
            if NUM_EPISODES  % 100 == 0:
            # 13. Obtain 全局共享的Q网络的状态
                self.Q.load_state_dict(self.shared_state["Q_state_dict"])



class Learner(object):
    def __init__(self, learner_params, shared_state):
        
        self.shared_state = shared_state
        self.env = Environment()
        self.params = learner_params
        self.shared_state = shared_state
        # if self.params['load_saved_state']:
        #     #尝试加载已经保存的训练好的模型
        #     try:
        #         saved_state = torch.load(self.params['load_saved_state'])
        #         current_model.load_state_dict(saved_state['Q_state'])
        #     except FileNotFoundError:
        #         print("WARNING: No trained model found. Training from scratch")

        #将当前的 Q 网络的参数保存在 shared_state["Q_state_dict"] 中
        self.Q = RainbowDQN(num_states, num_actions, num_atoms, Vmin, Vmax)
        self.Q_Double = RainbowDQN(num_states, num_actions, num_atoms, Vmin, Vmax)
        self.shared_state["Q_state_dict"] = self.Q.state_dict()
        #优化器RMSprop是常用的一种优化算法，它的基本思想是对梯度的历史信息进行加权平均，从而减小梯度更新的波动，提高学习的稳定性。
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=0.00025 / 4, weight_decay=0.95, eps=1.5e-7)
        self.num_q_updates = 0

    #计算损失函数及优先级，输入N_Step_Transition类型的经验列表。
    def compute_loss_and_priorities(self, xp_batch):

        #将多个经验转换为N步经验并存储在n_step_transitions中。
        n_step_transitions = N_Step_Transition(*zip(*xp_batch))
        # 将元组转换为numpy数组，将观测值(当前状态S_t and n步后的状态S_tpn) 转换为 torch Tensors 
        rew_t_to_tpB = np.array(n_step_transitions.R_ttpB)##n-step回报
        gamma_t_to_tpB = np.array(n_step_transitions.Gamma_ttpB)
        A_t = np.array(n_step_transitions.A_t,dtype=object)##当前动作


        rew_t_to_tpB = torch.from_numpy(rew_t_to_tpB).float()
        gamma_t_to_tpB = torch.from_numpy(gamma_t_to_tpB).float()


        S_t = torch.from_numpy(np.array(n_step_transitions.S_t)).float().requires_grad_(True)
        S_tpn = torch.from_numpy(np.array(n_step_transitions.S_tpn)).float().requires_grad_(True)
        S_tpn = np.array(n_step_transitions.S_tpn)
        #将S_tpn输入到目标模型（target_model）中，计算Q值，module的shape是torch.Size([5, 2, 51])
        #5是batch_size,2=actions，51=atoms
        with torch.no_grad():
            Q_double = self.Q_Double(torch.from_numpy(S_tpn).float())
        #Q_double_2 = Q_double[1]: 从Q值中提取第二维度的值，就是action对应的Q值
        Q_double_2 = Q_double[1]
       
        #将rew_t_to_tpB和gamma_t_to_tpB张量拼接在一起，形成一个新的张量
        rew_gamma = torch.cat((rew_t_to_tpB.unsqueeze(-1), gamma_t_to_tpB.unsqueeze(-1)), dim=-1)
        #计算Q_double_2的最大值和最大值的索引
        max_q_tp1, max_q_tp1_idx = Q_double_2.max(dim=1, keepdim=True)
        #将max_q_tp1张量从计算图中分离出来
        max_q_tp1 = max_q_tp1.detach()
        #将max_q_tp1_idx张量从计算图中分离出来。
        max_q_tp1_idx = max_q_tp1_idx.detach()
        #从max_q_tp1张量中删除一维（只需要最大Q值，而不需要知道它对应的动作是什么）
        max_q_tp1 = max_q_tp1.squeeze(-1)
        ##计算N步回报G_t，它是rew_gamma和gamma_t_to_tpB与max_q_tp1的乘积之和。
        G_t = rew_gamma + gamma * max_q_tp1
        G_t = G_t.numpy()
        print("G-T",G_t)
        #G_T_0是一维数组
        G_t_0 = G_t[:, 0]
        print("G_t0",G_t_0)
        G_t_0 = torch.from_numpy (G_t_0)#转换成tensor类型

        Q = self.Q(S_t)[1]
        Q = self.Q(S_t)[1].reshape(5,-1)
        print(Q)
        A_t = torch.from_numpy(A_t).reshape(-1, 1)

        Q_S_A = Q.gather(1, A_t).squeeze()
        print("Q_S_A",Q_S_A)
        
        
        G_t_0 = torch.from_numpy (G_t_0)
        batch_td_error = G_t_0 - Q_S_A
        print("batch_td_error",batch_td_error)
        #采用的是 MSE 损失函数。
        loss = 1/2 * (batch_td_error)**2
        print("loss",loss)


        
        # print("S_tpn",S_tpn.shape)
        # print("reward",rew_t_to_tpB.shape)
        # print("gamma",gamma_t_to_tpB.shape)
        # print("A_T",A_t)
        # print("S_T",S_t.shape)
        # print("cuu-module",current_model(S_t).shape)
        # print("smodule",target_model(S_tpn).shape)


        # with torch.no_grad():
        #     #计算N步回报G_t，self.Q(S_tpn)[2] 计算出n步之后的状态对应的Q值估计，.gather() 函数按照当前动作选择出对应的Q值估计
        #     #torch.argmax() 函数选择出最大的 Q值对应的动作索引，最后调用 squeeze() 函数将维度为 1 的维度压缩掉
        #     G_t = rew_t_to_tpB + gamma_t_to_tpB * \
        #                      target_model(S_tpn)[2].gather(1, torch.argmax(current_model(S_tpn)[2], 1).view(-1, 1)).squeeze()
        
        # 计算经验优先级
        #n_step_transitions.key 返回的是一个列表，包含了当前批量的所有转换数据的 key 值，
        # abs(batch_td_error.detach().data.numpy()) 返回的是当前批量的所有转换数据的 TD
        #priorities = {k: val for k in n_step_transitions.key for val in abs(batch_td_error.detach().data.numpy())}
        priorities = dict()
        for i in range(len(n_step_transitions.key)):
            priorities[n_step_transitions.key[i]]= batch_td_error[i]
        print("priorities", priorities)
        return loss.mean(), priorities

    def update_Q(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.num_q_updates += 1
        #更新网络的频率
        if self.num_q_updates % self.params['q_target_sync_freq']:
            self.Q_Double.load_state_dict(self.Q.state_dict())

    def learn(self):
        #当重放经验池的大小小于 self.params["min_replay_mem_size"] 时，
        # 等待回放记忆的大小达到指定大小。
        # time.sleep(1) 将使程序等待一秒钟，然后检查回放记忆的大小。
        while global_replay_queue.size() <=  self.params["min_replay_mem_size"]:
            time.sleep(1)

        T = 0
        for episode in range(NUM_EPISODES):
            
            start = time.time()
            total_reward = 0
            accepted_tasks=0
            #环境初始化
            self.env.reset()
            observation = self.env.observe()  
            #将当前状态从numpy数组转换为 PyTorch 张量
            state = observation  
            state = torch.from_numpy(state).type(
                torch.FloatTensor)  
            #对状态张量进行扩展，使state可以作为神经网络的输入
            state = torch.unsqueeze(state, 0)   
            #每个回合的循环
            for step in range(MAX_TASKS-1):
                #根据当前状态，从目标模型中获取行动
                action = self.Q_Double.act(state)
                #与环境交互，通过执行动作a_t 找到 s_{t+1},r_{t+1}
                observation_next, reward, done = self.env.update_env(
                    action.item()) 
                #给予奖励
                if done: 
                    next_state = None
                    break
                reward_t = torch.FloatTensor([reward]) #奖励值                    
                next_state = observation_next                     
                next_state = torch.from_numpy(next_state).type(
                         torch.FloatTensor)  
                next_state = torch.unsqueeze(next_state, 0)

                # 1优先级采样
                prioritized_xp_batch = global_replay_queue.sample(int(self.params['replay_sample_size']))
                # 2、3. 利用 target_module , 计算 loss 和 experience priorities
                loss, priorities = self.compute_loss_and_priorities(prioritized_xp_batch)
                #print("\nLearner: t=", NUM_EPISODES, "loss:", loss, "RPM.size:", global_replay_queue.size(), end='\r')
                
                #4. 将leaner的经验放进缓冲池
                #leaner_experience_batch = global_replay_queue.get(10) 
                global_replay_queue.add(priorities, prioritized_xp_batch)

                # 5. 更新 priorities
                global_replay_queue.set_priorities(priorities)
                
                
                
                # 5. 调用 update_Q() 函数来更新 Q 网络。将计算的损失作为参数传递。
                self.update_Q(loss)

                #6. 将 current_module 网络的状态字典存储在共享状态字典中。
                self.shared_state['Q_state_dict'] = self.Q.state_dict()
                
                # 更新观测值
                state = next_state

                # 7. Periodically remove old experience from replay memory
            if NUM_EPISODES % self.params['remove_old_xp_freq'] == 0:
                global_replay_queue.remove_to_fit()
                #target_model.load_state_dict(current_model.state_dict())

            end = time.time()
            times = end - start
            T = T + times
            print(" ")
            print(self.env.task)
            print(self.env.time_windows)
            state_next = None
            #record data
            total_reward = self.env.total_profit
            accepted_tasks= self.env.atasks
            eval_profit_list.append(total_reward)
            avg = sum(eval_profit_list)/len(eval_profit_list)
            avg_profit_list.append(avg)        
            eval_tasks_list.append(accepted_tasks)
            avg_1 = sum(eval_tasks_list)/len(eval_tasks_list)
            avg_tasks_list.append(avg_1)
            avg_profit = total_reward/accepted_tasks
            eval_ap_list.append(avg_profit)
            avg_2 = sum(eval_ap_list)/len(eval_ap_list)
            avg_ap_list.append(avg_2)
            print('%d Episode: Accepted tasks numbers：%d Total reward: %d'%(episode,accepted_tasks,total_reward))
        #print and draw
        print('Aveage Total Profit', avg)
        print('Aveage accepted tasks',avg_1)
        print('Aveage Profit', avg_2)
        print('Average Response Time', T / episode)
        plot_profit(eval_profit_list,avg_profit_list)
        plot_tasks(eval_tasks_list,avg_tasks_list)
        plot_ap(eval_ap_list,avg_ap_list)  




if __name__ == "__main__":
    """ 
    Simple standalone test routine for Actor class
    """
    env_conf = {"state_shape": 4,
                "action_dim": 2,
                "name": "Breakout-v0"}
    params = {"local_experience_buffer_capacity": 10,
              "epsilon": 0.4,
              "alpha": 7,
              "gamma": 0.99,
              "num_actors": 2,
              "n_step_transition_batch_size": 5,
              "Q_network_sync_freq": 10,
              "num_steps": 3,
              "remove_old_xp_freq": 100,
              "q_target_sync_freq": 100,
              "min_replay_mem_size": 2000,
              "replay_sample_size": 32,
              "T": 50  # Total number of time steps to gather experience

              }
    current_model = RainbowDQN(env_conf['state_shape'], env_conf['action_dim'],num_atoms, Vmin, Vmax)
    target_model = RainbowDQN(env_conf['state_shape'], env_conf['action_dim'],num_atoms, Vmin, Vmax)
    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_state["Q_state_dict"] = target_model.state_dict()
    shared_replay_mem = mp_manager.Queue()

    #actor = MultiActor(1, shared_state, params)
    leaner = Learner(params, shared_state)
    # learner_proc = mp.Process(target=leaner.learn)
    # learner_proc.start()

    leaner.learn()
    #actor.run()

    print("Main: mem.size:", global_replay_queue.size())
    print("Main: replay_mem.size:", shared_replay_mem.qsize())
    for i in range(shared_replay_mem.qsize()):
        p, xp_batch = shared_replay_mem.get()
        print("priority:", p)




















