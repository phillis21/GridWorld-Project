import numpy as np

class GridWorldEnv:
    def __init__(self,grid):
        pass
        self.grid = grid
        self.height, self.width = grid.shape
        self.start_pos = (0,0)
        self.goal_pos = (7,4)
        self.agent_pos = self.start_pos
        

    def move(self, direction,grid):
        new_x, new_y = self.agent_pos
        if direction == "up":
            new_x -= 1
        elif direction == "down":
            new_x += 1
        elif direction == "right":
            new_y += 1
        elif direction == "left":
            new_y -= 1

        if 0<= new_x < self.height and 0 <= new_y < self.width:
            if grid[new_x, new_y] != 'w':
                self.agent_pos = (new_x, new_y)
        else:
            print("Hit a wall")

    def render(self):
        for i in range(self.height):
            row = ''
            for j in range(self.width):
                if(i, j)== self.agent_pos:
                    row += 'A'
                elif self.grid[i, j] =='w':
                    row += 'X'
                elif self.grid[i, j] == 'g':
                    row += 'G'
                elif self.grid[i, j] == 's':
                    row += 'S'
                else:
                    row += ' . '
            print(row)
        print()


grid = np.array ([
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

env.render()

env.move('right',grid)
env.move('down',grid)
env.render()