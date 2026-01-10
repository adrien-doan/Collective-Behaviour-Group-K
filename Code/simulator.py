import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field
from utils.geometry import normalize
from utils.msr_utils import msr_filter
from agents.Dog import Dog
from obstacle import random_regular_polygon


# ---------- Parameters ----------

@dataclass
class Params:
    dt: float = 0.3

    # Sheep-sheep interaction radii
    rs: float = 10.0       
    r_sep: float = 3.0    

    # Sheep-dog radius
    rd: float = 20.0

    # Sheep gains
    Ks_sep: float = 4.5   # separation
    Ks_align: float = 0.01 # alignment
    Ks_coh: float = 0.75   # cohesion
    Ks_dog: float = 1.0   # dog avoidance

    sheep_vmax: float = 1.0

    sheep_rest_damping = 0.4
    stress_threshold = 0.001

    # Dog gains
    Kf_drive: float = 4.0
    dog_vmax: float = 2.5
    Kf_repulse: float = 25.0
    Kf_goal: float = 2.0
    Kf_sheep = 1.0      
    
    sd_radius = 20.0 # dog sheep vision radius 
    drive_soft_radius = 2.0

    # Goal
    goal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    goal_radius: float = 15.0

    # Dog "behind" distance from target sheep (relative to goal)
    drive_offset: float = 3.0

    # MSR param
    msr_enabled: bool = True
    msr_criterion: str = "g"    # "n", "b", or "g"
    msr_f: int = 1              # number of neighbors to discard in hard MSR

    # Obstacle
    K_obs = 1.0
    obstacle_radius = 5.0 #distance obstacle taken account of
    obstacle_pushback = 1.0

    steps: int = 1500

# ---------- Simulation ----------
class ShepherdSim:
    def __init__(self, n_sheep=80, n_dog = 1, n_abnormal_dog = 0, n_obstacle = 0, params: Params | None = None, seed=42):
        self.rng = np.random.default_rng(seed)
        self.params = params or Params()

        self.obstacles = []
        for i in range(n_obstacle):
            # choose random center in your world bounds
            cx = self.rng.uniform(-60, 60)
            cy = self.rng.uniform(-10, 50)
            center = np.array([cx, cy])

            obs = random_regular_polygon(
                rng=self.rng,
                center=center,
                min_radius= 5.0,
                max_radius= 12.0
            )
            self.obstacles.append(obs)

        # Initial sheep positions: a blob above the goal
        self.sheep_pos = np.empty((n_sheep, 2))
        self.sheep_pos[:, 0] = self.rng.uniform(-15, 15, size=n_sheep)
        self.sheep_pos[:, 1] = self.rng.uniform(30, 70, size=n_sheep)
        invalid_mask = self.sheep_inside_any_obstacle(self.sheep_pos)

        while np.any(invalid_mask):
            n_bad = invalid_mask.sum()

            self.sheep_pos[invalid_mask, 0] = self.rng.uniform(-15, 15, size=n_bad)
            self.sheep_pos[invalid_mask, 1] = self.rng.uniform(30, 70, size=n_bad)

            invalid_mask = self.sheep_inside_any_obstacle(self.sheep_pos)

        self.sheep_vel = self.rng.normal(scale=0.2, size=(n_sheep, 2))


        center = self.flock_center()
        dir_gc = center - self.params.goal          # direction goal -> flock
        dir_gc_norm = normalize(dir_gc[None, :])[0]

        self.dogs = []
        for i in range(n_dog - n_abnormal_dog):
            pos = center + dir_gc_norm * (self.params.drive_offset + 20 + 5*i)
            
            attempts = 0
            while not self.valid_point(pos):
                attempts += 1
                pos = center + dir_gc_norm * (self.params.drive_offset + 20 + 5*i + attempts * 5.0)

        # perpendicular to goal→flock direction
        perp = np.array([-dir_gc_norm[1], dir_gc_norm[0]])

        lateral_spacing = 8.0   # horizontal spacing between dogs
        back_offset     = 25.0  # how far behind the flock they start

        for i in range(n_dog - n_abnormal_dog):
            lateral_offset = (i - 0.5*(n_dog-1)) * lateral_spacing

            pos = (
                center
                + dir_gc_norm * (self.params.drive_offset + back_offset)
                + perp * lateral_offset
            )

            vel = np.zeros(2)
            self.dogs.append(Dog(pos, vel, self.params, dog_id=i))
    
        for i in range(n_abnormal_dog):
            pos = center + dir_gc_norm * (self.params.drive_offset + 20 + 5*i)
            vel = np.zeros(2)
            dog = Dog(pos, vel, self.params, dog_id=i + (n_dog - n_abnormal_dog), behavior= "wrong_goal")
            gap = 20
            dog.local_params["goal"] =np.array([40,-40])
            self.dogs.append(dog)

    # ----- Helper methods -----

    def valid_point(self, point):
        return not any(obs.contains(point) for obs in self.obstacles)

    def sheep_inside_any_obstacle(self, positions):
        """
        positions: (N,2) array
        returns: boolean mask of shape (N,) True if inside an obstacle
        """
        import numpy as np
        mask = np.zeros(positions.shape[0], dtype=bool)
        for obs in self.obstacles:
            # shapely works per-point → loop, but small
            mask |= np.array([obs.contains(pos) for pos in positions])
        return mask

    def flock_center(self):
        return self.sheep_pos.mean(axis=0)

    def farthest_sheep_from_goal_dir(self):
        """
        Find the sheep that is farthest along the line from the goal through the flock.
        That sheep is the one we want to push from behind.
        """
        g = self.params.goal
        center = self.flock_center()

        # Direction from goal toward flock center
        dir_gc = center - g
        dir_gc_norm = normalize(dir_gc[None, :])[0]

        # Project each sheep onto that direction (scalar projection)
        rel = self.sheep_pos - g
        proj = rel @ dir_gc_norm  # shape (n,)
        idx = int(np.argmax(proj))
        return self.sheep_pos[idx], idx, dir_gc_norm

    def all_sheep_in_goal(self):
        diffs = self.sheep_pos - self.params.goal
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        return np.all(d2 <= self.params.goal_radius ** 2)
    
    def compute_true_coverage(self, sheep_pos):
        """
        Compute proportion of sheep in goal
        Returns : coverage (float) between 0 and 1
        """
        p = self. params
        inside = np.linalg.norm(sheep_pos - p.goal, axis=1) < p.goal_radius
        coverage = inside.mean()
        return coverage

    # ----- Physics step -----

    def step(self):
        p = self.params
        n = self.sheep_pos.shape[0]

        # Pairwise differences: sheep_i - sheep_j
        diff_ss = self.sheep_pos[:, None, :] - self.sheep_pos[None, :, :]  # (n, n, 2)
        dist_ss = np.linalg.norm(diff_ss, axis=-1) + 1e-8                  # (n, n)

        neighbor_mask = (dist_ss < p.rs) & (dist_ss > 0.0)
        sep_mask = (dist_ss < p.r_sep) & (dist_ss > 0.0)

        # Separation: push away from very close neighbors (1/r^2)
        sep_dir = diff_ss / (dist_ss[..., None] ** 2)
        sep_dir *= sep_mask[..., None]
        n_sep = sep_mask.sum(axis=1, keepdims=True).clip(min=1)
        a_sep = sep_dir.sum(axis=1) / n_sep  # (n, 2)

        # Alignment: align with velocities of neighbors
        v_norm = normalize(self.sheep_vel)                       # (n, 2)
        vj_masked = v_norm[None, :, :] * neighbor_mask[:, :, None]
        n_nei = neighbor_mask.sum(axis=1, keepdims=True).clip(min=1)
        a_align = vj_masked.sum(axis=1) / n_nei                  # (n, 2)

        # Cohesion: pull towards neighbors (1/r)
        coh_dir = -diff_ss / dist_ss[..., None]
        coh_dir *= neighbor_mask[..., None]
        a_coh = coh_dir.sum(axis=1) / n_nei                      # (n, 2)

        # Dog avoidance: repulsion from dog (1/r^2, smoother than 1/r^3)
        
        dog_positions = np.array([dog.pos for dog in self.dogs])

        diff_s_dog = self.sheep_pos[:,None,:] - dog_positions[None,:,:]     # (N,M,2)
        dist_s_dog = np.linalg.norm(diff_s_dog, axis=-1, keepdims=True)
        dog_mask = (dist_s_dog < p.rd).astype(float)


        # Repulsion from all dogs → sum over dimension M
        a_dog = (diff_s_dog / (dist_s_dog)) * dog_mask
        a_dog = a_dog.sum(axis=1)

        a_obs = np.zeros_like(self.sheep_pos)

        for obs in self.obstacles:
            for i, pos in enumerate(self.sheep_pos):
                d, n = obs.distance_and_normal(pos)
                if d < p.obstacle_radius:
                    a_obs[i] += n / (d + 1e-8)**5

        # Total sheep acceleration
        sheep_acc = (
            p.Ks_sep   * a_sep +
            p.Ks_align * a_align +
            p.Ks_coh   * a_coh +
            p.Ks_dog   * a_dog +
            p.K_obs    * a_obs - 1e-1       * np.abs(self.sheep_vel) * self.sheep_vel
        )

        dog_pressure = np.linalg.norm(a_dog, axis=1)
        #print(dog_pressure)
        #low_stress = dog_pressure < p.stress_threshold
        #sheep_acc[low_stress] -= p.sheep_rest_damping * self.sheep_vel[low_stress]

        # Update sheep velocity and clamp speed
        self.sheep_vel += p.dt * sheep_acc
        speeds = np.linalg.norm(self.sheep_vel, axis=1, keepdims=True) + 1e-8
        too_fast = speeds > p.sheep_vmax
        self.sheep_vel[too_fast[:, 0]] *= (p.sheep_vmax / speeds[too_fast[:, 0]])

        # Update sheep positions
        self.sheep_pos += p.dt * self.sheep_vel

        # for obs in self.obstacles:
        #     for i in range(self.sheep_pos.shape[0]):
        #         pos = self.sheep_pos[i]
        #         if obs.contains(pos):
        #             d, n = obs.distance_and_normal(pos)
        #             # Push sheep out
        #             self.sheep_pos[i] += n * self.params.obstacle_pushback


        # ----- Dog dynamics -----

        coverages = []
        for dog in self.dogs:
            coverages.append(dog.compute_coverage(self.sheep_pos))

        trusted_coverage = msr_filter(coverages, p.msr_f)

        for dog in self.dogs:
            dog.trusted_coverage = trusted_coverage

        for dog in self.dogs:
            others = [d for d in self.dogs if d.id != dog.id]
            dog.step(self.sheep_pos, others, self.obstacles)

        proportion_sheep_in_goal = self.compute_true_coverage(self.sheep_pos)
        return {
            "all_in_goal": proportion_sheep_in_goal >= 0.95,
        }

    # ----- Run loop -----

    def run(self, max_steps=None, stop_when_goal=True):
        steps = max_steps or self.params.steps
        for k in range(steps):
            info = self.step()
            if stop_when_goal and info["all_in_goal"]:
                return k + 1
        return steps

# ---------- Animation ----------

def animate_run(n_sheep=200, n_dog = 3, n_abnormal_dog = 0, n_obstacles = 0, steps=1500, seed=4, interval_ms=1):
    if (n_abnormal_dog > n_dog):
        raise ValueError(" The number of abnormal can't be higher than the number of dogs")
    params = Params()
    sim = ShepherdSim(n_sheep=n_sheep, n_dog = n_dog,  n_abnormal_dog =  n_abnormal_dog, n_obstacle = n_obstacles, params=params, seed=seed)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Draw goal region
    circle = plt.Circle(params.goal, params.goal_radius, fill=False, linestyle="--", color="black")
    ax.add_artist(circle)


    g_list = [d.local_params["goal"] for d in sim.dogs if d.behavior == "wrong_goal"]
    for g in g_list:
        ax.scatter(g[0], g[1], color = "orange", marker= "x")

    sheep_sc = ax.scatter(sim.sheep_pos[:, 0], sim.sheep_pos[:, 1], s=10, label="Sheep")
    normal_dogs = []
    abnormal_dogs = []
    for d in sim.dogs:
        if d.behavior == "normal":
            normal_dogs.append(d)
        else:
            abnormal_dogs.append(d)
    
    dog_sc = ax.scatter(
        [d.pos[0] for d in normal_dogs],
        [d.pos[1] for d in normal_dogs],
        color="red",
        marker="*",
        s=120,
        label = "Dog"
    )
    a_dog_sc = ax.scatter(
        [d.pos[0] for d in abnormal_dogs],
        [d.pos[1] for d in abnormal_dogs],
        color="orange",
        marker="*",
        s=120,
        label = "Abnormal Dog"
    )

    # Draw obstacles
    for obs in sim.obstacles:
        poly = obs.vertices
        # close polygon for plotting
        poly_closed = np.vstack([poly, poly[0]])
        ax.plot(poly_closed[:,0], poly_closed[:,1], color="black", lw=2)
        ax.fill(poly_closed[:,0], poly_closed[:,1], color="gray", alpha=0.3)

    ttl = ax.set_title("Shepherding — step 0")
    ax.legend(loc="upper right")

    ax.set_xlim(-80, 60)
    ax.set_ylim(-50, 90)

    step_counter = {"k": 0}

    def frame_generator(sim, max_steps):
        for k in range(max_steps):
            info = sim.step()
            yield info
            if info["all_in_goal"]:
                break

    def _update(frame):
        info = sim.step()
        step_counter["k"] += 1

        sheep_sc.set_offsets(sim.sheep_pos)
        dog_sc.set_offsets([d.pos for d in normal_dogs])
        if(len(abnormal_dogs)):
            a_dog_sc.set_offsets([d.pos for d in abnormal_dogs])
        ttl.set_text(f"Shepherding — step {step_counter['k']}")

        return sheep_sc, dog_sc, ttl

    anim = animation.FuncAnimation(fig, _update, interval=interval_ms, blit=False, repeat=False,frames=frame_generator(sim, steps), cache_frame_data=False)

    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.split(".")[0].replace(" ", "_").replace(":","-")

    anim.save("results/shepherding_simulation-" + timestamp + ".gif", writer="pillow", fps=1000)
    plt.savefig("results/shepherding_simulation-" + timestamp + "final_step.png")
    #plt.show()
    plt.close(fig)

# Run the demo
if __name__ == "__main__":
    animate_run()
