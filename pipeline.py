import numpy as np
import networkx as nx
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable
from problems import GraphProblem
from policies import BasePolicy
from waypoints import c_waypoint, i_waypoint

def plan_trajectory(start: int, goal: int, env: GraphProblem, policy: BasePolicy, psi: Callable[[int], np.ndarray], A: np.ndarray,
                     waypoint_type: str, max_waypoints=100, max_steps=1000, M = 50, c = 1, eps=1e-3, T = 1) -> List[int]:

    env.reset(start, goal)

    curr_s = start

    trajectory = [curr_s]
    num_waypoints = 0

    if max_waypoints > 0:
        # choose between c-waypoint and i-waypoint
        if waypoint_type == 'c':
            waypoint = c_waypoint(start, goal, psi, A, list(env.graph.nodes()), M, T)
        elif waypoint_type == 'i':
            waypoint = i_waypoint(start, goal, psi, A, c)
        else:
            raise ValueError("Invalid waypoint type. Must be 'c' or 'i'.")
    else:
        waypoint = psi(goal)
    
    while curr_s != goal and len(trajectory) < max_steps:
        curr_z = A @ psi(curr_s)

        if np.linalg.norm(curr_z - waypoint) < eps:
            if num_waypoints < max_waypoints:
                num_waypoints += 1
                if waypoint_type == 'c':
                    waypoint = c_waypoint(curr_s, goal, psi, A, env.graph.nodes(), M)
                elif waypoint_type == 'i':
                    waypoint = i_waypoint(curr_s, goal, psi, A, c)
                else:
                    raise ValueError("Invalid waypoint type. Must be 'c' or 'i'.")
            else:
                waypoint = psi(goal)
        
        action = policy.get_action(curr_s, waypoint)
        trajectory.append(action)
        
        # step environment with action
        next_s, _ = env.step(action)
        curr_s = next_s
    
    return trajectory
            
        

