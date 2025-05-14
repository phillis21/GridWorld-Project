import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from collections import deque
import heapq

class GridWorldEnv:
    def __init__(self, grid):
        self.grid = grid
        self.height, self.width = grid.shape
        self.start_pos = self.find_tile('s')
        self.goal_pos = self.find_tile('g')
        self.agent_pos = self.start_pos

    def find_tile(self, tile):
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == tile:
                    return (i, j)
        return None

    def reset(self):
        self.agent_pos = self.start_pos

    def render(self, path=None, visited=None):
        cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red', 'blue', 'gray'])
        norm = mcolors.BoundaryNorm([0,1,2,3,4,5,6], cmap.N)

        render_grid = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 'w':
                    render_grid[i, j] = 1
                elif self.grid[i, j] == 'g':
                    render_grid[i, j] = 2
                elif self.grid[i, j] == 's':
                    render_grid[i, j] = 3

        if visited:
            for (x, y) in visited:
                if render_grid[x, y] == 0:
                    render_grid[x, y] = 5  # gray for visited

        if path:
            for (x, y) in path:
                if render_grid[x, y] == 0:
                    render_grid[x, y] = 4  # blue for path

        ax = plt.gca()
        ax.clear()
        ax.set_title("Grid World Environment")
        ax.imshow(render_grid, cmap=cmap, norm=norm)

        # Draw agent
        ax.plot(self.agent_pos[1], self.agent_pos[0], 'ro', markersize=12)

        plt.pause(0.2)

class Node:
    def __init__(self, position, parent=None, cost=0, heuristic=0):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

class Agent:
    def __init__(self, env):
        self.env = env

    def bfs(self):
        start = self.env.start_pos
        goal = self.env.goal_pos
        queue = deque()
        queue.append(Node(start))
        visited = set()
        visited.add(start)
        all_visited = set()

        while queue:
            current = queue.popleft()
            all_visited.add(current.position)

            if current.position == goal:
                return self.reconstruct_path(current), all_visited

            for neighbor in self.get_neighbors(current.position):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(Node(neighbor, parent=current))

        return None, all_visited

    def a_star(self):
        start = self.env.start_pos
        goal = self.env.goal_pos

        open_set = []
        heapq.heappush(open_set, Node(start, cost=0, heuristic=self.manhattan(start, goal)))
        visited = set()
        all_visited = set()

        while open_set:
            current = heapq.heappop(open_set)
            all_visited.add(current.position)

            if current.position == goal:
                return self.reconstruct_path(current), all_visited

            if current.position in visited:
                continue

            visited.add(current.position)

            for neighbor in self.get_neighbors(current.position):
                if neighbor not in visited:
                    cost = current.cost + 1
                    heuristic = self.manhattan(neighbor, goal)
                    heapq.heappush(open_set, Node(neighbor, parent=current, cost=cost, heuristic=heuristic))

        return None, all_visited

    def get_neighbors(self, pos):
        x, y = pos
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        neighbors = []

        for dx, dy in directions:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.env.height and 0 <= ny < self.env.width:
                if self.env.grid[nx, ny] != 'w':
                    neighbors.append((nx, ny))

        return neighbors

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        path.reverse()
        return path

    def move_along_path(self, path, visited=None):
        for step in path:
            self.env.agent_pos = step
            self.env.render(path, visited)

# === RUN ===

def main():
    grid = np.array([
        ['s','w',0,0,0,0,0,0],
        [0,0,0,0,0,'w',0,'w'],
        ['w',0,0,0,0,'w',0,0],
        ['w',0,0,0,0,'w',0,0],
        ['w',0,0,'w',0,0,0,'g'],
        ['w',0,'w',0,0,'w',0,0],
        ['w','w','w',0,'w','w','w',0],
        [0,0,0,0,0,0,0,0]
    ])

    env = GridWorldEnv(grid)
    agent = Agent(env)

    plt.ion()
    fig = plt.figure(figsize=(6,6))

    print("Solving with BFS...")
    path, visited = agent.bfs()
    if path:
        print("Path found:", path)
        agent.move_along_path(path, visited)
    else:
        print("No path found with BFS.")

    time.sleep(2)

    env.reset()

    print("Solving with A*...")
    path, visited = agent.a_star()
    if path:
        print("Path found:", path)
        agent.move_along_path(path, visited)
    else:
        print("No path found with A*.")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
