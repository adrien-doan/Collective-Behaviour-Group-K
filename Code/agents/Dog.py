import numpy as np
from utils.geometry import normalize

class Dog:
    def __init__(self, pos, vel, params, dog_id):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.params = params
        self.id = dog_id
    
    def select_target_sheep(self, sheep_pos, goal):

        center = sheep_pos.mean(axis=0)
        dir_gc = center - goal
        dir_gc_norm = normalize(dir_gc[None])[0]

        rel = sheep_pos - goal
        proj = rel @ dir_gc_norm
        idx = int(np.argmax(proj))
        return idx, dir_gc_norm

    def compute_dog_repulsion(self, other_dogs_pos):
        diff = self.pos - other_dogs_pos
        dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8
        rep = diff / (dist**2)
        return rep.sum(axis=0)
    
    def compute_sheep_repulsion(self, sheep_pos):
        diff = self.pos - sheep_pos               # shape (N,2)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        
        mask = (dist < self.params.sd_radius).astype(float)
        
        rep = (diff / (dist**2)) * mask
        return rep.sum(axis=0)

    def step(self, sheep_pos, other_dogs):
        p = self.params

        #  Target sheep
        target_idx, dir_gc_norm = self.select_target_sheep(sheep_pos, p.goal)
        target_pos = sheep_pos[target_idx]

        #Compute drive point 
        drive_point = target_pos + dir_gc_norm * p.drive_offset
        desired = drive_point - self.pos


        a_drive = desired / (np.linalg.norm(desired) + 1e-8)
        dist = np.linalg.norm(desired)
        w = np.tanh(dist / p.drive_soft_radius)

        a_drive = normalize(desired[None])[0] * w

        #Sheep repulsion 

        a_sheep = self.compute_sheep_repulsion(sheep_pos)

        # Inter-dog repulsion 
        if len(other_dogs) > 0:
            rep = self.compute_dog_repulsion(np.array([d.pos for d in other_dogs]))
        else:
            rep = np.zeros(2)

        #Goal repulsion for dogs
        goal_diff = self.pos - p.goal
        dist_g = np.linalg.norm(goal_diff) + 1e-8
        a_goal = -goal_diff / (dist_g**2)



        # Combined control
        acc = p.Kf_drive * a_drive + p.Kf_sheep * a_sheep + p.Kf_repulse * rep +  p.Kf_goal * a_goal

        # Integrate 
        self.vel += p.dt * acc
        speed = np.linalg.norm(self.vel)
        if speed > p.dog_vmax:
            self.vel *= p.dog_vmax / speed

        self.pos += p.dt * self.vel
