import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field

# ---------- Utility ----------

def normalize(vecs, eps=1e-8):
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / (norms + eps)

# ---------- Parameters ----------

@dataclass
class Params:
    dt: float = 0.3

    # Sheep-sheep interaction radii
    rs: float = 10.0       # neighbor radius
    r_sep: float = 2.0    # strong separation radius

    # Sheep-dog radius
    rd: float = 20.0

    # Sheep gains
    Ks_sep: float = 1.8   # separation
    Ks_align: float = 0.8 # alignment
    Ks_coh: float = 0.35   # cohesion
    Ks_dog: float = 1.2   # dog avoidance

    sheep_vmax: float = 1.1

    # Dog gains
    Kf_drive: float = 4.0
    dog_vmax: float = 3

    # Goal
    goal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    goal_radius: float = 10.0

    # Dog "behind" distance from target sheep (relative to goal)
    drive_offset: float = 5.0

    steps: int = 1500

# ---------- Simulation ----------

class ShepherdSim:
    def __init__(self, n_sheep=80, params: Params | None = None, seed=42):
        self.rng = np.random.default_rng(seed)
        self.params = params or Params()

        # Initial sheep positions: a blob above the goal
        self.sheep_pos = np.empty((n_sheep, 2))
        self.sheep_pos[:, 0] = self.rng.uniform(-15, 15, size=n_sheep)
        self.sheep_pos[:, 1] = self.rng.uniform(30, 70, size=n_sheep)
        self.sheep_vel = self.rng.normal(scale=0.2, size=(n_sheep, 2))

        # Dog starts below the flock
        self.dog_vel = np.zeros(2, dtype=float)

        center = self.flock_center()
        dir_gc = center - self.params.goal          # direction goal -> flock
        dir_gc_norm = normalize(dir_gc[None, :])[0]

        # Put the dog further along that direction (behind the flock)
        self.dog_pos = center + dir_gc_norm * (self.params.drive_offset + 20.0)
    # ----- Helper methods -----

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
        dog_diff = self.sheep_pos - self.dog_pos                 # (n, 2)
        dog_dist = np.linalg.norm(dog_diff, axis=1, keepdims=True) + 1e-8
        dog_mask = (dog_dist < p.rd).astype(float)
        a_dog = (dog_diff / (dog_dist ** 2)) * dog_mask          # (n, 2)

        # Total sheep acceleration
        sheep_acc = (
            p.Ks_sep   * a_sep +
            p.Ks_align * a_align +
            p.Ks_coh   * a_coh +
            p.Ks_dog   * a_dog
        )

        # Update sheep velocity and clamp speed
        self.sheep_vel += p.dt * sheep_acc
        speeds = np.linalg.norm(self.sheep_vel, axis=1, keepdims=True) + 1e-8
        too_fast = speeds > p.sheep_vmax
        self.sheep_vel[too_fast[:, 0]] *= (p.sheep_vmax / speeds[too_fast[:, 0]])

        # Update sheep positions
        self.sheep_pos += p.dt * self.sheep_vel

        # ----- Dog dynamics -----

        # Find "front-most" sheep along goal→flock direction
        target_pos, target_idx, dir_gc_norm = self.farthest_sheep_from_goal_dir()

        # Desired drive point: behind that sheep relative to the goal
        drive_point = target_pos + dir_gc_norm * p.drive_offset

        # Dog accelerates toward drive point
        desired = drive_point - self.dog_pos
        a_drive = normalize(desired[None, :])[0]

        dog_acc = p.Kf_drive * a_drive

        self.dog_vel += p.dt * dog_acc
        dog_speed = np.linalg.norm(self.dog_vel) + 1e-8
        if dog_speed > p.dog_vmax:
            self.dog_vel *= (p.dog_vmax / dog_speed)

        self.dog_pos += p.dt * self.dog_vel

        return {
            "all_in_goal": self.all_sheep_in_goal(),
            "target_idx": target_idx,
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

def animate_run(n_sheep=80, steps=300, seed=4, interval_ms=1):
    params = Params()
    sim = ShepherdSim(n_sheep=n_sheep, params=params, seed=seed)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Draw goal region
    circle = plt.Circle(params.goal, params.goal_radius, fill=False, linestyle="--", color="black")
    ax.add_artist(circle)

    sheep_sc = ax.scatter(sim.sheep_pos[:, 0], sim.sheep_pos[:, 1], s=10, label="Sheep")
    dog_sc = ax.scatter([sim.dog_pos[0]], [sim.dog_pos[1]], marker="*", s=120, label="Dog", color="red")
    ttl = ax.set_title("Shepherding — step 0")
    ax.legend(loc="upper right")

    ax.set_xlim(-70, 40)
    ax.set_ylim(-20, 90)

    step_counter = {"k": 0}

    def _update(frame):
        info = sim.step()
        step_counter["k"] += 1

        sheep_sc.set_offsets(sim.sheep_pos)
        dog_sc.set_offsets(sim.dog_pos[None, :])
        ttl.set_text(f"Shepherding — step {step_counter['k']}")

        if info["all_in_goal"] or step_counter["k"] >= steps:
            anim.event_source.stop()
        return sheep_sc, dog_sc, ttl

    anim = animation.FuncAnimation(fig, _update, interval=interval_ms, blit=False, repeat=False,frames=270)
    anim.save("shepherding_simulation2.gif", writer="pillow", fps=1000)
    plt.show()
    plt.close(fig)

# Run the demo
if __name__ == "__main__":
    animate_run()
