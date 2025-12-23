import numpy as np
from utils.geometry import normalize
from utils.msr_utils import sim_b_heading, sim_g_goal, sim_n_proximity
from obstacle import Obstacle 

class Dog:
    def __init__(self, pos, vel, params, dog_id, behavior = "normal"):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.params = params
        self.id = dog_id
        self.behavior = behavior
        self.trusted_coverage = 0

        self.local_params = {
            "Kf_drive": params.Kf_drive,
            "Kf_repulse": params.Kf_repulse,
            "Kf_sheep": params.Kf_sheep,
            "Kf_goal": params.Kf_goal,
            "goal": params.goal.copy(),
        }
        
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
    
    def compute_trusted_indices(self, other_dogs, criterion="b", f=1):
        """
        criterion: "n", "b", or "g"
        returns: list of trusted Dog objects
        """
        M = len(other_dogs)
        if M == 0:
            return []

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
        return trusted_dogs
    
    def compute_obstacle_repulsion(self, obstacles):
        total = np.zeros(2)
        for obs in obstacles:
            d, n = obs.distance_and_normal(self.pos)
            if d < self.params.obstacle_radius:
                total += self.params.K_obs * (1.0 / (d + 1e-8)**3) * n
        return total
    
    def compute_coverage(self, sheep_pos):
        """
        Compute proportion of sheep in goal
        Returns : coverage (float) between 0 and 1
        """
        lp = self.local_params
        p = self. params
        inside = np.linalg.norm(sheep_pos - lp["goal"], axis=1) < p.goal_radius
        coverage = inside.mean()
        return coverage


    def step(self, sheep_pos, other_dogs, obstacles):
        p = self.params
        lp = self.local_params

        #  Target sheep
        target_idx, dir_gc_norm = self.select_target_sheep(sheep_pos, lp["goal"])
        target_pos = sheep_pos[target_idx]

        #Compute drive point 
        drive_point = target_pos + dir_gc_norm * p.drive_offset
        desired = drive_point - self.pos


        a_drive = desired / (np.linalg.norm(desired) + 1e-8)
        dist = np.linalg.norm(desired)
        w = np.tanh(dist / p.drive_soft_radius)

        a_drive = normalize(desired[None])[0] * w

        if self.trusted_coverage > 0.95:
            a_drive *= 0.5

        #Sheep repulsion 
        a_sheep = self.compute_sheep_repulsion(sheep_pos)

        # Inter-dog repulsion 
        if len(other_dogs) > 0:
            # MSR
            if(p.msr_enabled):
                trusted = self.compute_trusted_indices(other_dogs, criterion=p.msr_criterion, f=p.msr_f)
                if len(trusted) > 0:
                    rep = self.compute_dog_repulsion(np.array([d.pos for d in trusted]))
                else:
                    rep = np.zeros(2)
            else:
                self.compute_dog_repulsion(np.array([d.pos for d in other_dogs]))
        else:
            rep = np.zeros(2)

        #Goal repulsion for dogs
        goal_diff = self.pos - lp["goal"]
        dist_g = np.linalg.norm(goal_diff) + 1e-8
        a_goal = goal_diff / (dist_g)

        # Obstacle repulsion
        a_obs = self.compute_obstacle_repulsion(obstacles)

        # Combined control
        acc = lp["Kf_drive"] * a_drive + lp["Kf_sheep"] * a_sheep + lp["Kf_repulse"] * rep +  lp["Kf_goal"] * a_goal + p.K_obs * a_obs

        # Integrate 
        self.vel += p.dt * acc
        self.vel *= (1 - 0.3 * self.trusted_coverage) 
        speed = np.linalg.norm(self.vel)
        if speed > p.dog_vmax:
            self.vel *= p.dog_vmax / speed

        self.pos += p.dt * self.vel

        for obs in obstacles:
            if obs.contains(self.pos):
                # push dog to nearest boundary point
                _, normal = obs.distance_and_normal(self.pos)
                self.pos -= normal * self.params.obstacle_pushback

