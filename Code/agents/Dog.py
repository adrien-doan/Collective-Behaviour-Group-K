import numpy as np
from utils.geometry import normalize
from utils.msr_utils import sim_b_heading, sim_g_goal, sim_n_proximity

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
          # extra hard push if very close
        close = dist < 4.0
        rep[close[:, 0]] *= 5.0
        return rep.sum(axis=0)
    
    def compute_sheep_repulsion(self, sheep_pos):
        diff = self.pos - sheep_pos               # shape (N,2)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        
        mask = (dist < self.params.sd_radius).astype(float)
        
        rep = (diff / (dist**2)) * mask
        return rep.sum(axis=0)
    
    def compute_trusted_indices(self, other_dogs, criterion="n", f=1):
        """
        criterion: "n", "b", or "g"
        returns: list of trusted Dog objects
        """
        M = len(other_dogs)
        if M == 0:
            return []
        
        return other_dogs

        others_pos = np.array([d.pos for d in other_dogs])
        others_vel = np.array([d.vel for d in other_dogs])

        if criterion == "n":
            scores = sim_n_proximity(self.pos, others_pos)
        elif criterion == "b":
            scores = sim_b_heading(self.vel, others_vel)
        elif criterion == "g":
            scores = sim_g_goal(self.pos, others_pos, self.params.goal)
        else:
            raise ValueError("Unknown criterion")

        # sort indices by descending score (most similar first)
        idx_sorted = np.argsort(-scores)
        # number to keep:
        keep = max(0, M - f)
        if keep <= 0:
            # fallback: keep all
            trusted_idx = idx_sorted
        else:
            trusted_idx = idx_sorted[:keep]
        trusted_dogs = [other_dogs[i] for i in trusted_idx]
        print(f"Dog {self.id} trusts dogs {[d.id for d in trusted_dogs]} based on criterion '{criterion}'")
        
        return [trusted_dogs]

    def step(self, sheep_pos, other_dogs):
        p = self.params

        #  Target sheep
        target_idx, dir_gc_norm = self.select_target_sheep(sheep_pos, p.goal)
        target_pos = sheep_pos[target_idx]

        #Compute drive point 
        # drive_point = target_pos + dir_gc_norm * p.drive_offset
        # perpendicular left/right direction
        perp = np.array([-dir_gc_norm[1], dir_gc_norm[0]])

        # spread dogs in a fan formation
        lateral_spacing = 6.0
        lateral_offset = (self.id - 0.5*(len(other_dogs))) * lateral_spacing

        drive_point = (
            target_pos
            + dir_gc_norm * p.drive_offset
            + perp * lateral_offset
        )
        desired = drive_point - self.pos


        a_drive = desired / (np.linalg.norm(desired) + 1e-8)
        dist = np.linalg.norm(desired)
        w = np.tanh(dist / p.drive_soft_radius)

        a_drive = normalize(desired[None])[0] * w

        #Sheep repulsion 
        a_sheep = self.compute_sheep_repulsion(sheep_pos)

        # Inter-dog repulsion 
        if len(other_dogs) > 0:
            # MSR
            trusted = self.compute_trusted_indices(other_dogs, criterion=p.msr_criterion, f=p.msr_f)
            if len(trusted) > 0:
                rep = self.compute_dog_repulsion(np.array([d.pos for d in trusted]))
            else:
                rep = np.zeros(2)
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
