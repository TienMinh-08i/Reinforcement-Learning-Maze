from generate_maze import Maze
import numpy as np
import random
import matplotlib.pyplot as plt
import pygame

class MazeEnv:
    def __init__(self, maze, cell_size=20):
        self.maze = maze
        self.agent_pos = maze.startNode  # Vị trí khởi đầu của agent
        self.goal = maze.sinkerNode  # Vị trí đích
        self.total_steps = 0  # Đếm số bước đã đi

        # Kích thước cho mỗi ô vuông trong mê cung
        self.cell_size = cell_size
        self.width = (maze.width * 2 - 1) * self.cell_size
        self.height = (maze.height * 2 - 1) * self.cell_size

        # Khởi tạo pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Maze Environment')

    def reset(self):
        """Khởi tạo lại trạng thái ban đầu của môi trường"""
        self.agent_pos = self.maze.startNode
        self.total_steps = 0
        return self.agent_pos

    def step(self, action):
        """Thực hiện hành động và nhận lại trạng thái mới, phần thưởng, và trạng thái kết thúc"""
        possible_actions = self.get_possible_actions()
        reward = -0.1  # Phần thưởng tiêu cực cho mỗi bước di chuyển để khuyến khích tìm đường nhanh nhất

        if action in possible_actions:
            self.agent_pos = action  # Cập nhật vị trí mới của agent
        else:
            reward = -1  # Nếu hành động không hợp lệ, phần thưởng tiêu cực lớn hơn

        done = False
        if self.agent_pos == self.goal:
            reward = 10  # Phần thưởng khi đến đích
            done = True  # Kết thúc nếu agent đến đích

        self.total_steps += 1
        return self.agent_pos, reward, done

    def get_possible_actions(self):
        """Lấy các hành động có thể từ vị trí hiện tại của tác nhân"""
        current_node = self.agent_pos
        possible_actions = []
        for i, connected in enumerate(self.maze.adjacency[current_node]):
            if connected > 0:
                possible_actions.append(i)
        return possible_actions

    def get_action_at_position(self, position):
        """Lấy các hành động có thể từ một vị trí cụ thể"""
        current_node = position
        possible_actions = []
        for i, connected in enumerate(self.maze.adjacency[current_node]):
            if connected > 0:
                possible_actions.append(i)
        return possible_actions

    def render(self):
        """Hiển thị mê cung và vị trí của tác nhân trong pygame"""
        self.screen.fill((0, 0, 0))  # Xóa màn hình với màu trắng

        maze_map = self.maze.generate_maze_map()

        for y in range(maze_map.shape[0]):
            for x in range(maze_map.shape[1]):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if maze_map[y, x] == 1:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)  # Màu đen cho tường

        # Vị trí bắt đầu
        start_x, start_y = self.maze.node2xy(self.maze.startNode)
        start_rect = pygame.Rect(start_x * self.cell_size, start_y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 0, 255), start_rect)  # Màu xanh cho điểm bắt đầu

        # Vị trí đích
        goal_x, goal_y = self.maze.node2xy(self.maze.sinkerNode)
        goal_rect = pygame.Rect(goal_x * self.cell_size, goal_y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), goal_rect)  # Màu đỏ cho điểm đích

        # Vị trí hiện tại của agent
        agent_x, agent_y = self.maze.node2xy(self.agent_pos)
        agent_rect = pygame.Rect(agent_x * self.cell_size, agent_y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), agent_rect)  # Màu xanh lá cho vị trí hiện tại của agent

        pygame.display.flip()  # Cập nhật màn hình

    def close(self):
        pygame.quit()

def run_rl_episode(env):
    """Chạy một tập (episode) trong môi trường RL"""
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

        env.render()  # Hiển thị mê cung và tác nhân ở mỗi bước

        # Chọn một hành động ngẫu nhiên
        possible_actions = env.get_possible_actions()
        action = random.choice(possible_actions)

        # Thực hiện hành động
        next_state, reward, done = env.step(action)
        total_reward += reward

        print(f"Agent moved to {next_state}, Reward: {reward}, Total steps: {env.total_steps}")
        pygame.time.delay(500)

    print(f"Episode finished. Total reward: {total_reward}, Total steps: {env.total_steps}")
    pygame.time.delay(5000)  # Dừng lại 2 giây trước khi kết thúc

if __name__ == "__main__":
    myMaze = Maze().load(r"C:\Users\admin\Desktop\mazee\maze_8x8.pkl")
    env = MazeEnv(myMaze)
    run_rl_episode(env)
    env.close()