import numpy as np

eps = 1e-9

def sim_n_proximity(self_pos, others_pos):
    # higher = more similar (closer)
    d = np.linalg.norm(others_pos - self_pos, axis=1)
    return 1.0 / (d + eps) 

def sim_b_heading(self_vel, others_vel):
    # compute cosine similarity between headings
    def safe_norm(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (n + eps)
    s_self = (self_vel / (np.linalg.norm(self_vel) + eps))[None, :] 
    s_others = safe_norm(others_vel)                               
    return (s_others @ s_self.T).reshape(-1) 

def sim_g_goal(self_pos, others_pos, goal):
    d_self = np.linalg.norm(self_pos - goal)
    d_others = np.linalg.norm(others_pos - goal, axis=1)
    return 1.0 / (np.abs(d_others - d_self) + eps)   # higher = similar distance
