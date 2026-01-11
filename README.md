# Group K — Replicating and Extending “Guidance by Multiple Sheepdogs Including Abnormalities”

## Team Members
| Name | GitHub Username |
|------|------------------|
| Adrien Doan | [@adrien-doan](https://github.com/adrien-doan) |
| Edita Džubur | [@eddzu](https://github.com/eddzu) |

---

## Project Description

This project is part of the **Collective Behavior** course.  
Our objective is to **replicate and extend** the model described in:

> Tashiro, M., Kubo, M., Sato, H., & Yamaguchi, A. (2022).  
> *Guidance by multiple sheepdogs including abnormalities*.  
> *Artificial Life and Robotics*, 27, 714–725.  
> [https://doi.org/10.1007/s10015-022-00807-1](https://doi.org/10.1007/s10015-022-00807-1)

In this study, a flock of autonomous sheep agents (**A-sheep**) is guided by multiple sheepdog agents (**s-dogs**) using repulsive forces.  
The paper introduces the **MSR (Mean Subsequence Reduced)** algorithm to maintain robust coordination even when some sheepdogs behave abnormally (e.g., sensor faults or conflicting goals).

Our project will:
- Reproduce this model computationally,
- Evaluate it under various conditions (normal and abnormal dogs),
- Propose and test potential **improvements** to the model.

---

## Starting Point

We begin from the mathematical model and experimental design provided in the paper.  
The first step will be to implement the following components:

- **A-sheep model:** flocking dynamics based on attraction, alignment, and repulsion.  
- **s-dog model:** single and multiple dog behaviors with mutual repulsion.  
- **MSR-dog agents:** versions that incorporate anomaly-tolerant cooperation (n-dog, b-dog, g-dog).  
- **Simulation environment:** a 2D field where sheep and dogs interact and move toward a goal.

---

##  Project Plan and Milestones

### ✅ Milestone 1 — Completion Summary

#### What We Achieved

- We went through the main literature on flocking and shepherding to understand the background of the problem.
- We studied the Tashiro et al. (2022) model in detail, including how sheep and dogs behave in the system and how abnormal dogs are handled.
- We built the first version of the simulation, where a single dog guides a flock using the basic rules from the paper.
- We produced some early visual outputs showing the flock moving and the dog steering it toward the goal.
- We wrote and formatted the first report, added figures, and organised the document into a clear structure.
- We set up the project repository and outlined the next steps for Milestone 2.

#### Team Contributions

Adrien contributed by gathering the relevant literature, reviewing existing shepherding and collective-behaviour work, and preparing the initial outline and draft of the report. Edita focused on the implementation, building the first version of the flocking and single-dog system, and then refining and completing the written report.

---


###  Milestone 2 — Second Report (Due **December 7, 2025**)

**Objectives**
- Extend the model to **multiple sheepdogs** and implement repulsion-based coordination.  
- Integrate the **MSR algorithm** to handle abnormal dog behavior.  
- Replicate some key results and figures from the original paper.  
- Evaluate baseline vs. MSR models for performance and stability.  

**Deliverables**
- Working multi-dog simulation with and without anomalies.  
- Preliminary performance results (success rate, guidance time).  
- Updated report with visualizations and discussion.

---

###  Milestone 3 — Final Report (Due **January 11, 2026**)

**Objectives**
- Extend or improve the model with one or more new features:
  - Simulate obstacles and realistic dog vision fields.  
  - Make repulsion adaptive to flock density.  
  - Introduce stamina or energy constraints for dogs.  
- Perform a comprehensive analysis of all models.  
- Compare our results to the baseline and MSR implementations.  
- Prepare the final report and presentation.  

**Deliverables**
- Complete repository with code, experiments, and documentation.  
- Final analytical report.  
- Updated `README.md` summarizing final results and insights.

---

##  Expected Results

- **Multiple dogs** will outperform single-dog systems by achieving spatial distribution around the flock.  
- The **MSR algorithm** will improve resilience against abnormal agents (e.g., faulty or hacked sheepdogs).  
- **Abnormalities** will degrade performance, but cooperative filtering (b-dog, g-dog strategies) will mitigate these effects.  
- Larger groups of dogs will enhance stability and robustness.

---

##  Repository Usage

This repository is the central place for our code, documentation, and reports.  
It also tracks our progress through commits, branches, issues, and milestones.

#### Repository structure
- Code: Python simulation code
- Literature: All referenced papers used in the report
- Report: LaTeX source, images and the final compiled PDF

#### Running simulation
1. Open `Code/` folder.
2. Make sure Python (3.12.7 or newer) is installed along with `numpy` and `matplotlib` packages.
3. Run main script: `python simulator.py`

Parameters regarding the number of sheep, dogs, abnormal dogs, obstacles and steps can be modified in the function 'animate_run' in the file 'Code/simulator.py'.

Example :

```
def animate_run(n_sheep=200, n_dog = 3, n_abnormal_dog = 0, n_obstacles = 0, steps=1500, seed=4, interval_ms=1):
```


Parameters regarding the movement gains, forces, range and MSR type can be modified in the class 'Params' in the file 'Code/simulator.py'.




