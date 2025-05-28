import copy
import time
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from collections import deque
import heapq
from pyswip import Prolog

# env/base 
env_grid = np.array([
    ['s','w',0,   0,   0,   0,   0, 0],
    [0,   0,  0,   'w', 0,   0,   0,   0],
    [0,  'w', 0,   'w', 0,  'w',  0,   'g'],
    [0,   0,  0,   0,   0,  'w',  0,   0],
    [0,  'w', 0,   0,   0,   0,  'w',  0],
    [0,   0,  0,  0,  0,   0,   0,   0],
    ['w', 0 ,'w', 0 ,'w','w','w', 'w'],
    [0,   0,  0,   0,   0,   0,   0,   0]
])

class GridWorldEnv:
    def __init__(self, grid):
        self.grid = grid
        self.height, self.width = grid.shape
        self.start_pos = self._find_tile('s')
        self.goal_pos  = self._find_tile('g')
        self.agent_pos = self.start_pos

    def _find_tile(self, tile):
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i,j] == tile:
                    return (i,j)
        return None

    def reset(self):
        self.agent_pos = self.start_pos

    def render(self, path=None, visited=None):
        cmap = mcolors.ListedColormap(['white','black','green','red','blue','gray','yellow'])
        render_grid = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i,j] == 'w': render_grid[i,j] = 1
                elif self.grid[i,j] == 'g': render_grid[i,j] = 2
                elif self.grid[i,j] == 's': render_grid[i,j] = 3
        ax = plt.gca(); ax.clear()
        if visited:
            for x,y in visited:
                if render_grid[x,y] == 0:
                    render_grid[x,y] = 5
        if path:
            for x,y in path:
                render_grid[x,y] = 4
        ax.imshow(render_grid, cmap=cmap, vmin=0, vmax=6)
        ax.plot(self.agent_pos[1], self.agent_pos[0], 'ro', markersize=12)
        legend_handles = [
            Patch(color='gray', label='Cells explored'),
            Patch(color='blue', label='Final path')
        ]
        ax.legend(handles=legend_handles, loc='upper right')
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
    def bfs(self, target=None):
        start = self.env.agent_pos
        goal = target if target else self.env.goal_pos
        queue = deque([Node(start)])
        visited = {start}
        all_visited = set()
        while queue:
            current = queue.popleft()
            all_visited.add(current.position)
            if current.position == goal:
                return self.reconstruct_path(current), all_visited
            for n in self.get_neighbors(current.position):
                if n not in visited:
                    visited.add(n)
                    queue.append(Node(n, parent=current))
        return None, all_visited
    def a_star(self):
        start = self.env.start_pos; goal = self.env.goal_pos
        open_set=[]; heapq.heappush(open_set, Node(start,0,self.manhattan(start,goal)))
        visited, all_v=set(), set()
        while open_set:
            cur=heapq.heappop(open_set)
            all_v.add(cur.position)
            if cur.position==goal:
                return self.reconstruct_path(cur), all_v
            if cur.position in visited: continue
            visited.add(cur.position)
            for n in self.get_neighbors(cur.position):
                if n not in visited:
                    cost=cur.cost+1
                    heapq.heappush(open_set, Node(n,cur,cost,self.manhattan(n,goal)))
        return None, all_v
    def get_neighbors(self,pos):
        dirs=[(-1,0),(1,0),(0,-1),(0,1)]; res=[]
        x,y=pos
        for dx,dy in dirs:
            nx,ny=x+dx,y+dy
            if 0<=nx<self.env.height and 0<=ny<self.env.width and self.env.grid[nx,ny] != 'w':
                res.append((nx,ny))
        return res
    def manhattan(self,a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    def reconstruct_path(self,node):
        p=[]
        while node: p.append(node.position); node=node.parent
        return p[::-1]
    def move_along_path(self,path,visited=None):
        for s in path:
            self.env.agent_pos = s
            self.env.render(path, visited)

class MultiAgentEnv(GridWorldEnv):
    def __init__(self,grid,positions): super().__init__(grid); self.agent_positions=positions
    def render(self,path=None,visited=None):
        super().render(path,visited)
        ax=plt.gca(); marks=['ro','go','bo','mo']
        for i,pos in enumerate(self.agent_positions): ax.plot(pos[1],pos[0],marks[i%len(marks)],markersize=12)
        plt.pause(0.2)
    def step(self,aid,p): self.agent_positions[aid]=p


class CompetitiveAgent(Agent):
    def minimax(self,env,aid,depth,maxi):
        p0,p1=env.agent_positions
        if p0==p1 or depth==0: return self.utility(p0,p1),None
        best_val=(-1e9 if maxi else 1e9); best_m=None
        cur=env.agent_positions[aid]
        for mv in self.get_neighbors(cur):
            ec=copy.deepcopy(env); ec.step(aid,mv)
            val,_=self.minimax(ec,1-aid,depth-1,not maxi)
            if (maxi and val>best_val) or (not maxi and val<best_val): best_val,best_m=val,mv
        return best_val,best_m
    def utility(self,p,q): return -self.manhattan(p,q)

# prolog
class PrologAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.prolog = Prolog()
        self._write_prolog_kb()
        self.prolog.consult('prolog_rules.pl')

    def _write_prolog_kb(self):
        kb = '''
path((SX,SY),(GX,GY),Path) :- path_helper((SX,SY),(GX,GY),[(SX,SY)],Rev), reverse(Rev,Path).
path_helper((X,Y),(X,Y),V,V).
path_helper((X,Y),(GX,GY),V,P) :-
    neighbor(X,Y,NX,NY),
    \+ member((NX,NY),V),
    path_helper((NX,NY),(GX,GY),[(NX,NY)|V],P).
'''
        with open('prolog_rules.pl','w') as f:
            f.write(kb)
        # Assert cell facts
        for i in range(self.env.height):
            for j in range(self.env.width):
                tp = 'wall' if self.env.grid[i,j]=='w' else 'free'
                self.prolog.assertz(f"cell({i},{j},{tp})")
        # Assert neighbor facts
        for i in range(self.env.height):
            for j in range(self.env.width):
                if self.env.grid[i,j] != 'w':
                    for nx,ny in self.get_neighbors((i,j)):
                        self.prolog.assertz(f"neighbor({i},{j},{nx},{ny})")

    def plan_with_prolog(self):
        s, g = self.env.start_pos, self.env.goal_pos
        results = list(self.prolog.query(f"path(({s[0]},{s[1]}),({g[0]},{g[1]}),P)"))
        return results[0]['P'] if results else None

# forage
class ForagingEnv(GridWorldEnv):
    def __init__(self, grid, food_positions):
        super().__init__(grid)
        self.food = set(food_positions)
    def render(self, path=None, visited=None):
        cmap = mcolors.ListedColormap(['white','black','green','red','blue','gray','yellow'])
        render_grid = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i,j]=='w': render_grid[i,j]=1
                elif (i,j) in self.food: render_grid[i,j]=6
                elif self.grid[i,j]=='s': render_grid[i,j]=3
        if visited:
            for x,y in visited:
                if render_grid[x,y]==0: render_grid[x,y]=5
        if path:
            for x,y in path: render_grid[x,y]=4
        ax=plt.gca(); ax.clear(); ax.set_title('Foraging Environment')
        ax.imshow(render_grid,cmap=cmap,vmin=0,vmax=6)
        ax.plot(self.agent_pos[1], self.agent_pos[0], 'ro', markersize=12)
        lh=[Patch(color='yellow',label='Food'), Patch(color='gray',label='Explored'), Patch(color='blue',label='Path')]
        ax.legend(handles=lh, loc='upper right')
        plt.pause(0.2)

class ForagerAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
    def forage(self):
        collected=0
        while self.env.food:
            pos = self.env.agent_pos
            # find nearest food
            target = min(self.env.food, key=lambda f: self.manhattan(pos, f))
            # use A* to plan to target
            original_start = self.env.start_pos
            original_goal = self.env.goal_pos
            self.env.start_pos = pos
            self.env.goal_pos = target
            path, visited = self.a_star()
            # restore original start/goal
            self.env.start_pos = original_start
            self.env.goal_pos = original_goal
            if not path:
                break
            self.move_along_path(path, visited)
            self.env.food.remove(target)
            collected += 1
        print(f"Collected {collected} food pellets.")

# runner functions 
def run_start_to_goal():
    """Start-to-goal navigation with choice of BFS, A*, or Prolog."""
    planner_menu={'1':'BFS','2':'A*','3':'Prolog'}
    print("\nSelect planner for start-to-goal:")
    for k,v in planner_menu.items(): print(f"[{k}] {v}")
    choice=input("Planner choice: ").strip()

    env=GridWorldEnv(env_grid)
    plt.ion(); plt.figure(figsize=(6,6))
    path=visited=None
    if choice=='1': 
        agent=Agent(env); 
        path,visited=agent.bfs()
    elif choice=='2': 
        agent=Agent(env); 
        path,visited=agent.a_star()
    elif choice=='3': 
        agent=PrologAgent(env); 
        raw=agent.plan_with_prolog() or []; visited=None
        path=[]
        for t in raw:
            nums=re.findall(r"-?\d+", str(t))
            if len(nums)>=2: path.append((int(nums[0]), int(nums[1])))
    else:
        print("Invalid choice."); return
    if path:
        for step in path: env.agent_pos=step; env.render(path, visited)
    else: print("No path found.")
    plt.ioff(); time.sleep(10); plt.close('all')


def run_phase2():
    env=MultiAgentEnv(env_grid,[(0,0),(7,7)])
    agent=CompetitiveAgent(env)
    plt.ion(); plt.figure(figsize=(6,6))
    for t in range(50):
        _,mv=agent.minimax(env,t%2,3,True)
        if not mv: break
        env.step(t%2,mv); env.render()
        if env.agent_positions[0]==env.agent_positions[1]: break
    plt.ioff(); time.sleep(10); plt.close('all')


def run_foraging():
    """Foraging with choice of BFS, A*, or Prolog planning."""
    food=[(1,2),(2,4),(5,0),(7,7),(3,5)]
    env=ForagingEnv(env_grid,food)
    planner_menu={'1':'BFS','2':'A*','3':'Prolog'}
    print("\nSelect planner for foraging:")
    for k,v in planner_menu.items(): print(f"[{k}] {v}")
    choice=input("Planner choice: ").strip()
    if choice=='3': prolog_agent=PrologAgent(env)
    plt.ion(); plt.figure(figsize=(6,6))
    collected=0
    while env.food:
        pos=env.agent_pos
        best=None; 
        best_path=[]; 
        best_vis=None; 
        best_t=None
        for f in list(env.food):
            if choice=='1': 
                path,vis=Agent(env).bfs(f)
            elif choice=='2': 
                og=env.goal_pos; 
                env.goal_pos=f; 
                path,vis=Agent(env).a_star(); 
                env.goal_pos=og
            elif choice=='3': 
                env.start_pos=pos; 
                env.goal_pos=f; 
                raw=prolog_agent.plan_with_prolog() or []; vis=None; 
                path=[]
                for t in raw:
                    nums=re.findall(r"-?\d+",str(t))
                    if len(nums)>=2: path.append((int(nums[0]),int(nums[1])))
            else: print("Invalid."); return
            if path:
                l=len(path)
                if best is None or l<best: best=l; best_path=path; best_vis=vis; best_t=f
        if best is None: print("No reachable food."); break
        for step in best_path: env.agent_pos=step; env.render(best_path,best_vis)
        env.food.remove(best_t); collected+=1
    print(f"Collected {collected} food pellets.")
    plt.ioff(); time.sleep(10); plt.close('all')

# menu
if __name__=='__main__':
    menu={'1':'Start-to-goal','2':'Predator-Prey','3':'Foraging','4':'Exit'}
    while True:
        print("\nTasks:")
        for k,v in menu.items(): print(f"[{k}] {v}")
        c=input("Choice: ").strip()
        if c=='1': run_start_to_goal()
        elif c=='2': run_phase2()
        elif c=='3': run_foraging()
        elif c=='4': print("Goodbye"); break
        else: print("Invalid. Please choose 1-4.")
