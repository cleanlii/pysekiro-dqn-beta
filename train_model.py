# train_model.py

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import deque

from dqn_models import DoubleDQN
from get_status import get_game_status, get_self_HP, ROI
from grab_screen import get_game_screen
from get_keys import key_check
from direct_keys import lock, attack

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class RewardSystem:
    def __init__(self):
        self.total_reward = 0    # 当前积累的 reward
        self.reward_history = list()    # reward 的积累过程

    # 获取奖励
    def get_reward(self, cur_status, next_status):
        """
        cur_status 和 next_status 都是存放状态信息的列表，内容：[状态1, 状态2, 状态3, 状态4]
        cur_status  表示当前的人物状态
        next_status 表示未来的人物状态
        """
        if sum(next_status) == 0:
            reward = 0
        else:
            # 通过列表索引的方式，取出相应的信息，用未来的状态信息减去当前的状态信息，得到状态变化值
            # 自身生命值减少则为负，目标生命值减少则为正，架势值则刚好相反
            s1 = next_status[0] - cur_status[0] # 自身生命变化值
            s2 = next_status[1] - cur_status[1] # 自身架势变化指
            s3 = next_status[2] - cur_status[2] # 目标生命变化值
            s4 = next_status[3] - cur_status[3] # 目标架势变化值
            # s5 = 0

            # 定义得分
            # 雨露均沾型，可通过拔高s的值训练重点技能

            s1 *=  1    # 与 奖励 呈正相关，所以 +
            s2 *= -1    # 与 惩罚 呈正相关，所以 -
            s3 *= -1    # 与 惩罚 呈正相关，所以 -
            s4 *=  1    # 与 奖励 呈正相关，所以 +

            # 完美防御：生命值不减的同时，目标架势值上升，给予双倍奖励
            # if(s1 == 0 and s3 == 0 and s4 > 0):
            #     s5 = s4*2

            reward = s1 + s2 + s3 + s4

        self.total_reward += reward
        self.reward_history.append(self.total_reward)

        return reward

    def save_reward_curve(self, save_path='Log/reward.png'):
        total = len(self.reward_history)
        if total > 100:
            plt.rcParams['figure.figsize'] = 100, 15
            plt.plot(np.arange(total), self.reward_history)
            plt.ylabel('reward')
            plt.xlabel('training steps')
            plt.xticks(np.arange(0, total, int(total/100)))
            plt.savefig(save_path)
            plt.show()

# 200 x 200 window mode
x   = 800 // 2 - 200    # 左 不小于0，小于 x_w
x_w = 800 // 2 + 200    # 右 不大于图像宽度，大于 x
y   = 450 // 2 - 200    # 上 不小于0，小于 y_h
y_h = 450 // 2 + 200    # 下 不大于图像高度，大于 y

in_depth    = 1     # 卷积核，二维卷积为1
in_height   = 50    # 图像高度，图像缩放用
in_width    = 50    # 图像宽度，图像缩放用
in_channels = 1     # 颜色通道数量
outputs = 4     # 动作数量，即智能体能够执行几种动作
lr = 0.001      # 学习率，默认为0.001，如果要修改的话，前期可以设置大点，快速收敛，后期设置小一点，提升学习效果

gamma = 0.99    # 奖励衰减，未来对现在的重要程度，设置为 1 代表同等重要，模型更有远瞻性；设置的越小说明越重视当前的决策
replay_memory_size = 10000    # 记忆容量
replay_start_size = 500       # 开始经验回放时存储的记忆量，到达最终探索率后才开始
batch_size = 16               # 样本抽取数量
update_freq = 200                   # 训练评估网络的频率
target_network_update_freq = 500    # 更新目标网络的频率


class Agent:
    def __init__(
        self,
        save_memory_path=None,
        load_memory_path=None,
        save_weights_path=None,
        load_weights_path=None
    ):
        self.save_memory_path = save_memory_path     # 指定记忆/经验保存的路径。默认为None，不保存。
        self.load_memory_path = load_memory_path     # 指定记忆/经验加载的路径。默认为None，不加载。
        self.brain = DoubleDQN(
            in_depth,
            in_height,      # 图像高度
            in_width,       # 图像宽度
            in_channels,    # 颜色通道数量
            outputs,        # 动作数量
            lr,             # 学习率
            gamma,    # 奖励衰减
            replay_memory_size,     # 记忆容量
            replay_start_size,      # 开始经验回放时存储的记忆量，到达最终探索率后才开始
            batch_size,             # 样本抽取数量
            update_freq,                   # 训练评估网络的频率
            target_network_update_freq,    # 更新目标网络的频率
            save_weights_path,    # 指定模型权重保存的路径。默认为None，不保存。
            load_weights_path     # 指定模型权重加载的路径。默认为None，不加载。
        )
        if not save_weights_path:    # 注：默认也是测试模式，若设置该参数，就会开启训练模式
            self.train = False
            self.brain.step = self.brain.replay_start_size + 1
        else:
            self.train = True

        self.reward_system = RewardSystem()

        self.i = 0    # 计步器

        self.death = False # 判断死亡

        self.screens = deque(maxlen = in_depth * 2)    # 用双端队列存放图像

        if self.load_memory_path:
            self.load_memory()    # 加载记忆/经验

    def load_memory(self):
        if os.path.exists(self.load_memory_path):
            last_time = time.time()
            self.brain.replayer.memory = pd.read_json(self.load_memory_path)    # 从json文件加载记忆/经验。 
            print(f'Load {self.load_memory_path}. Took {round(time.time()-last_time, 3):>5} seconds.')

            i = self.brain.replayer.memory.action.count()
            self.brain.replayer.i = i
            self.brain.replayer.count = i
            self.brain.step = i

        else:
            print('No memory to load.')

    def get_S(self):

        for _ in range(in_depth):
            self.screens.append(get_game_screen())    # 先进先出，右进左出

    def img_processing(self, screens):
        return np.array([cv2.resize(ROI(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY), x, x_w, y, y_h), (in_height, in_width)) for screen in screens])

    def round(self):

        observation = self.img_processing(list(self.screens)[:in_depth])    # S

        action = self.action = self.brain.choose_action(observation)    # A

        self.get_S()    # 观测

        reward = self.reward_system.get_reward(
            cur_status=get_game_status(list(self.screens)[in_depth - 1]),
            next_status=get_game_status(list(self.screens)[in_depth * 2 - 1])
        )    # R

        next_observation = self.img_processing(list(self.screens)[in_depth:])    # S'

        if self.train:

            self.brain.replayer.store(
                observation,
                action,
                reward,
                next_observation
            )

            if self.brain.replayer.count >= self.brain.replay_start_size:
                self.brain.learn()

    def run(self):
        last_time = time.time()
        print('Training is ready!')
        print('Press T to start!')
        for i in list(range(5))[::-1]:
            print(i+1)
            time.sleep(1)

        paused = True

        while True:

            last_time = time.time()
            
            keys = key_check()
            
            if paused:
                if 'T' in keys:
                    self.get_S()
                    paused = False
                    print('\nStarting the train!')

            if not paused:

                self.i += 1

                self.round()

                print(f'\r {self.brain.who_play:>4} , step: {self.i:>6} . Loop took {round(time.time()-last_time, 3):>5} seconds. action {self.action:>1} , total_reward: {self.reward_system.total_reward:>10.3f} , memory: {self.brain.replayer.count:7>} .', end='')
 
                if 'P' in keys:
                    if self.train:
                        self.brain.save_evaluate_network()    # 学习完毕，保存网络权重
                        self.brain.replayer.memory.to_json(self.save_memory_path)    # 保存经验
                    self.reward_system.save_reward_curve()    # 绘制 reward 曲线并保存在当前目录
                    break

        print('\nTraining is done!')