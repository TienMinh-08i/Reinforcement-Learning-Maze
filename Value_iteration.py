import numpy as np
import random
import matplotlib.pyplot as plt
from env import MazeEnv
from generate_maze import Maze
import pygame
import pickle

import numpy as np

class MDP:
    def __init__(self, states, actions, transition_probs, rewards):
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs
        self.rewards = rewards

    def get_possible_actions(self, state):
        return self.actions

    def get_transition_prob(self, state, action, next_state):
        return self.transition_probs.get((state, action, next_state), 0.0)

    def get_reward(self, state, action, next_state):
        return self.rewards.get((state, action, next_state), 0.0)

def value_iteration(mdp, gamma=0.9, theta=0.0001):
    """
    Thuật toán Value Iteration để tính toán giá trị tối ưu cho mỗi trạng thái.
    """
    V = {s: 0 for s in mdp.states}  # Khởi tạo giá trị ban đầu cho tất cả các trạng thái là 0
    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            max_action_value = float('-inf')
            for a in mdp.get_possible_actions(s):
                action_value = 0
                for s_prime in mdp.states:
                    transition_prob = mdp.get_transition_prob(s, a, s_prime)
                    reward = mdp.get_reward(s, a, s_prime)
                    action_value += transition_prob * (reward + gamma * V[s_prime])
                max_action_value = max(max_action_value, action_value)

            V[s] = max_action_value
            delta = max(delta, np.abs(v - V[s]))

        if delta < theta:
            break

    return V

def extract_policy(mdp, V, gamma=0.9):
    """
    Trích xuất chính sách tối ưu từ bảng giá trị tối ưu.
    """
    policy = {}
    for s in mdp.states:
        best_action = None
        best_action_value = float('-inf')
        for a in mdp.get_possible_actions(s):
            action_value = 0
            for s_prime in mdp.states:
                transition_prob = mdp.get_transition_prob(s, a, s_prime)
                reward = mdp.get_reward(s, a, s_prime)
                action_value += transition_prob * (reward + gamma * V[s_prime])
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = a

        policy[s] = best_action
    return policy

def run_agent_with_optimal_policy(env, optimal_policy):
    """Chạy agent trong môi trường bằng cách sử dụng optimal policy."""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        env.render()

        # Chọn hành động dựa trên optimal policy
        action = optimal_policy.get(state)
        if action is None:
            print(f"Không tìm thấy hành động cho trạng thái {state}. Chọn hành động ngẫu nhiên.")
            possible_actions = env.get_possible_actions()
            if not possible_actions:
                print("Không có hành động nào có thể thực hiện được.")
                break
            action = random.choice(possible_actions)

        # Thực hiện hành động và cập nhật
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

        print(f"Agent moved to {next_state}, Reward: {reward}, Total steps: {steps}")
        pygame.time.delay(500)

    print(f"Episode finished. Total reward: {total_reward}, Total steps: {steps}")
    pygame.time.delay(5000)


myMaze = Maze().load(r"C:\Users\admin\Desktop\mazee\maze_4x4.pkl")
env = MazeEnv(myMaze)
# Ví dụ: Định nghĩa MDP đơn giản cho mê cung
states = list(range(myMaze.total_nodes))  # Danh sách các trạng thái
actions = list(range(myMaze.total_nodes))  # Danh sách các hành động
transition_probs = {}  # Xác suất chuyển đổi giữa các trạng thái
rewards = {}  # Phần thưởng cho mỗi chuyển đổi

for s in states:
    possible_actions = env.get_action_at_position(s)
    for a in possible_actions:
        transition_probs[(s, a, a)] = 1.0  # Giả sử chuyển đổi thành công với xác suất 1
        rewards[(s, a, a)] = -1.0 if a != myMaze.sinkerNode else 10.0  # Phần thưởng tiêu cực cho mỗi bước, thưởng tích cực cho đích



mdp = MDP(states, actions, transition_probs, rewards)
# optimal_values = value_iteration(mdp)
# optimal_policy = extract_policy(mdp, optimal_values)

# print("Optimal Values:")
# for s, v in optimal_values.items():
#     print(f"State {s}: Value = {v}")

# print("\nOptimal Policy:")
# for s, a in optimal_policy.items():
#     print(f"State {s}: Action = {a}")

# # Lưu policy vào file
# with open(r"C:\Users\admin\Desktop\mazee\VI_policy_maze_16x16.pkl", 'wb') as f:
#     pickle.dump(optimal_policy, f)

# Lấy policy từ file
with open(r"C:\Users\admin\Desktop\mazee\VI_policy_maze_16x16.pkl", 'rb') as f:
    loaded_optimal_policy = pickle.load(f)


# Chạy agent với optimal policy
run_agent_with_optimal_policy(env, loaded_optimal_policy)