import numpy as np
import matplotlib.pyplot as plt
from random import randrange, shuffle
import pickle
import datetime


def generate_adjacency_matrix(w, h):

    adjacency = np.zeros([h * w, h * w], dtype='float64')
    visited = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]

    def pos2node(x, y):
        return int(x * h + y)

    def walk(x, y):
        visited[y][x] = 1

        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        shuffle(d)
        for (xx, yy) in d:
            if visited[yy][xx]:
                continue
            if xx == x:
                adjacency[pos2node(x, yy), pos2node(x, y)] = 1
                adjacency[pos2node(x, y), pos2node(x, yy)] = 1
            if yy == y:
                adjacency[pos2node(x, y), pos2node(xx, y)] = 1
                adjacency[pos2node(xx, y), pos2node(x, y)] = 1

            walk(xx, yy)

    walk(randrange(w), randrange(h))

    return adjacency


class Maze(object):

    def __init__(self, adjacency=None, maze_size=(20, 10), startNode=0, sinkerNode=None):
        self.maze_size = maze_size
        self.width = int(self.maze_size[0])
        self.height = int(self.maze_size[1])

        if adjacency is None:
            self.adjacency = generate_adjacency_matrix(self.width, self.height)
        else:
            assert self.maze_size == adjacency.shape
            self.adjacency = adjacency

        self.total_nodes = self.width * self.height
        self.startNode = startNode
        self.sinkerNode = sinkerNode

        self.vertical_links = (self.height - 1) * self.width
        self.horizontal_links = (self.width - 1) * self.height
        self.total_links = self.vertical_links + self.horizontal_links

    @property
    def startNode(self):
        return self._startNode

    @startNode.setter
    def startNode(self, value):
        if value is None:
            self._startNode = 0
        else:
            assert 0 <= value < self.total_nodes, "sinkerNode is outside the node range"
            self._startNode = value

    @property
    def sinkerNode(self):
        return self._sinkerNode

    @sinkerNode.setter
    def sinkerNode(self, value):
        if value is None:
            self._sinkerNode = self.width * self.height - 1
        else:
            assert 0 <= value < self.total_nodes, "sinkerNode is outside the node range"
            self._sinkerNode = value

    def generate_maze_map(self):

        maze_map = np.zeros([2 * self.height - 1, 2 * self.width - 1], dtype='int')
        for j in range(self.width - 1):  # defines the upper triangular part of the adjacency matrix
            for i in range(self.height - 1):
                if self.adjacency[j * self.height + i, j * self.height + i + 1] > 0:
                    maze_map[2 * i + 1, 2 * j] = 1
                if self.adjacency[j * self.height + i, (j + 1) * self.height + i] > 0:
                    maze_map[2 * i, 2 * j + 1] = 1
            i = self.height - 1  # last node has no upper neighbour
            if self.adjacency[j * self.height + i, (j + 1) * self.height + i] > 0:
                maze_map[2 * i, 2 * j + 1] = 1

        j = self.width - 1  # last node has no right neighbour
        for i in range(self.height - 1):
            if self.adjacency[j * self.height + i, j * self.height + i + 1] > 0:
                maze_map[2 * i + 1, 2 * j] = 1

        for n in range(self.width * self.height):
            x, y = self.node2xy(n)
            maze_map[y, x] = 1

        return maze_map

    def plot_maze(self, show_nodes=False, show_links=False, show_ticks=False, show=True):

        maze_map = self.generate_maze_map()
        xshift = 0.33
        yshift = 0.33
        cmap = plt.colormaps.get_cmap('gray')
        norm = plt.Normalize(maze_map.min(), maze_map.max())
        img = cmap(norm(maze_map))

        if self.startNode is not None:
            x, y = self.node2xy(self.startNode)
            img[y, x, :3] = 0, 0, 1
            # plt.text(x - xshift, y - yshift, str(self.startNode), fontweight='bold') # always print startNode

        if self.sinkerNode is not None:
            x, y = self.node2xy(self.sinkerNode)
            img[y, x, :3] = 1, 0, 0
            # plt.text(x - xshift, y - yshift, str(self.sinkerNode), fontweight='bold') # always print sinkerNode

        if show:
            if show_nodes:
                for n in range(self.height * self.width):
                    x, y = self.node2xy(n)
                    plt.text(x - xshift, y - yshift, str(n), fontweight='bold')

            if show_links:
                for n in range(1, (self.height - 1) * self.width + (self.width - 1) * self.height + 1):
                    x, y = self.link2xy(n)
                    plt.text(x, y - yshift, str(n), style='italic', color='red')

            if show_ticks:
                plt.xticks(np.arange(0, img.shape[1], step=4),
                           np.arange(0, (img.shape[1] - 1) / 2, step=2, dtype='int'))
                plt.yticks(np.arange(0, img.shape[0], step=4),
                           np.arange(0, (img.shape[0] - 1) / 2, step=2, dtype='int'))
            else:
                plt.xticks([])
                plt.yticks([])

            ax = plt.imshow(img, origin='lower')
            plt.show()

        else:
            ax = None

        return img, ax

    def set_link(self, link=None, value=None):

        assert 1 <= link <= self.total_links
        if 1 <= link <= self.vertical_links:
            row = (link - 1) // (self.height - 1) * self.height + (link - 1) % (
                    self.height - 1)
            col = row + 1

        elif self.vertical_links < link <= self.total_links:
            row = link - (self.height - 1) * self.width - 1
            col = row + self.height

        self.adjacency[row, col] = value
        self.adjacency[col, row] = value

        return value

    def get_link(self, link=None):

        assert 1 <= link <= self.total_links
        if 1 <= link <= self.vertical_links:
            row = (link - 1) // (self.height - 1) * self.height + (link - 1) % (
                    self.height - 1)
            col = row + 1

        elif self.vertical_links < link <= self.total_links:
            row = link - (self.height - 1) * self.width - 1
            col = row + self.height

        return self.adjacency[row, col]

    def reverse_link(self, link=None):

        value = self.get_link(link)
        if value == 0:
            self.set_link(link, value=np.float64(1))
            return np.float64(1)
        elif value > 0:
            self.set_link(link, value=np.float64(0))
            return np.float64(0)

    def xy2node(self, x, y):

        assert 0 <= x <= 2 * self.width + 1
        assert 0 <= y <= 2 * self.height + 1
        if x % 2 == 0 and y % 2 == 0:
            n = int((x // 2) * self.height + y // 2)
        else:
            n = np.nan

        return n

    def node2xy(self, n):

        return 2 * (n // self.height), 2 * (n % self.height)

    def xy2link(self, x, y):

        if 0 <= x <= (self.width - 1) * 2 and x % 2 == 0:
            # vertical link
            n = int((x // 2) * (self.height - 1) + (y - 1) // 2 + 1)

        elif 0 <= y <= (self.height - 1) * 2 and y % 2 == 0:
            # horizontal link
            n = (x - 1) // 2 * self.height + y // 2 + 1
            n = int(n + (self.height - 1) * self.width)

        else:  # invalid parameters
            n = np.nan

        return n

    def link2xy(self, n):

        if 1 <= n <= self.vertical_links:  # vertical link
            x, y = 2 * ((n - 1) // (self.height - 1)), 2 * ((n - 1) % (self.height - 1)) + 1

        elif self.vertical_links < n <= self.total_links:  # horizontal link
            n = n - self.vertical_links
            x, y = 2 * ((n - 1) // self.height) + 1, 2 * ((n - 1) % self.height)

        else:  # invalid parameters
            x, y = np.nan, np.nan

        return x, y

    def save(self, filename=None):

        if filename is None:
            filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_maze'
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

        return filename

    def load(self, filename):


        assert filename is not None, "filename parameter in Maze.load() is None"

        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)

        self.__dict__.update(tmp_dict)

        return self


if __name__ == "__main__":
    print('maze_tools has started')
    # maze_tools test
    myMaze = Maze(maze_size=(16, 16))
    myMaze.plot_maze(show_nodes=False, show_links=False)
    myMaze.save(r"C:\Users\admin\Desktop\mazee\maze_16x16.pkl")
    #myMaze = None
    #myMaze = Maze().load(r"C:\Users\admin\Desktop\mazee\maze_4x4.pkl")
    #myMaze.plot_maze(show_nodes=False, show_links=False)
    # comparison test between myMaze and myMaze2
    #np.all([np.all(myMaze.__dict__[x] == myMaze2.__dict__[x]) for x in myMaze.__dict__.keys()])