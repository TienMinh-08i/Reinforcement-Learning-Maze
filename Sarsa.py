import numpy as np
import random
import matplotlib.pyplot as plt
from env import MazeEnv
from generate_maze import Maze
import pygame
import pickle

class SarsaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.maze.total_nodes, env.maze.total_nodes))

    def choose_action(self, state):
        """Chọn hành động dựa trên chính sách epsilon-greedy"""
        if random.uniform(0, 1) < self.epsilon:
            possible_actions = self.env.get_possible_actions()
            return random.choice(possible_actions)
        else:
            possible_actions = self.env.get_possible_actions()
            q_values = [self.q_table[state, a] for a in possible_actions]
            return possible_actions[np.argmax(q_values)]

    def learn(self, state, action, reward, next_state, next_action):
        """Cập nhật bảng Q bằng thuật toán SARSA"""
        td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

def run_sarsa(env, agent, episodes=1000):
    """Chạy nhiều episode bằng thuật toán SARSA"""
    total_rewards = []  # Danh sách để lưu tổng reward của mỗi episode
    best_reward = float('-inf')  # Biến theo dõi reward cao nhất
    best_q_table = None  # Biến lưu lại Q-table có reward cao nhất
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0  # Khởi tạo đếm bước cho mỗi episode

        # Chọn hành động ban đầu
        action = agent.choose_action(state)

        while not done:
            #env.render()

            # Thực hiện hành động
            next_state, reward, done = env.step(action)

            # Chọn hành động tiếp theo
            next_action = agent.choose_action(next_state)

            # Cập nhật bảng Q
            agent.learn(state, action, reward, next_state, next_action)

            # Chuyển sang trạng thái tiếp theo và hành động tiếp theo
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1  # Cập nhật số bước

            print(f"Agent moved to {next_state}, Reward: {reward}, Total steps: {env.total_steps}")

            # Kiểm tra số bước, nếu vượt quá 2000 thì kết thúc và trừ thêm reward
            if steps >= 5000:
                total_reward -= 10  # Trừ reward
                print("Số bước vượt quá giới hạn 2000, kết thúc episode với reward -10.")
                done = True  # Kết thúc episode

        total_rewards.append(total_reward)

        # Cập nhật reward cao nhất và Q-table nếu total_reward cao hơn
        if total_reward > best_reward:
            best_reward = total_reward
            best_q_table = np.copy(agent.q_table)  # Lưu lại Q-table của agent

        print(f"Episode {episode + 1}/{episodes} - Total reward: {total_reward}")

    return best_reward, best_q_table, total_rewards


myMaze = Maze().load(r"C:\Users\admin\Desktop\mazee\maze_16x16.pkl")
env = MazeEnv(myMaze)
agent = SarsaAgent(env)
total_reward, q_table, total_rewards = run_sarsa(env, agent, episodes=10000) # Modified line to unpack 3 values

# Lưu q_table vào file
with open(r"C:\Users\admin\Desktop\mazee\q_table_sarsa_maze_16x16.pkl", 'wb') as f:
    pickle.dump(q_table, f)

# Lấy q_table từ file
with open(r"C:\Users\admin\Desktop\mazee\q_table_sarsa_maze_16x16.pkl", 'rb') as f:
    loaded_q_table = pickle.load(f)


# Hàm vẽ đồ thị total_reward sau mỗi episode
def plot_rewards(total_rewards):
    plt.plot(total_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

plot_rewards(total_rewards) # Call the function to plot the rewards

# chạy agent theo q table vừa tạo ra
def run_agent_with_sarsa_q_table(env, q_table):
    """Chạy agent trong môi trường bằng cách sử dụng Q-table."""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        env.render()

        # Chọn hành động có giá trị Q cao nhất
        possible_actions = env.get_possible_actions()
        q_values = [q_table[state, a] for a in possible_actions]
        action = possible_actions[np.argmax(q_values)]

        # Thực hiện hành động và cập nhật
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

        print(f"Agent moved to {next_state}, Reward: {reward}, Total steps: {steps}")
        pygame.time.delay(500)

    print(f"Episode finished. Total reward: {total_reward}, Total steps: {steps}")
    pygame.time.delay(5000)

# Chạy agent với Q-table đã học
run_agent_with_sarsa_q_table(env, loaded_q_table)