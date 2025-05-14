import numpy as np

class QLearning(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减系数
        self.sample_count = 0
        self.Q_table = np.zeros((state_dim, action_dim))  # Q表格

    def choose_action(self, state):
        self.sample_count += 1
        # 动态调整epsilon（必须添加epsilon_min和epsilon_decay的定义）
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)  # 探索
        else:
            action = np.argmax(self.Q_table[state])  # 利用
        return action

    def update(self, state, action, reward, next_state, done):
        current_q = self.Q_table[state][action]
        target_q = reward + (1 - int(done)) * self.gamma * np.max(self.Q_table[next_state])
        self.Q_table[state][action] += self.lr * (target_q - current_q)

        # 添加predict方法（关键修复）

    def predict(self, state):
       return np.argmax(self.Q_table[state])  # 直接选择最优动作

    def save(self, path):
        np.save(path + "Q_table.npy", self.Q_table)

    def load(self, path):
        self.Q_table = np.load(path + "Q_table.npy")
