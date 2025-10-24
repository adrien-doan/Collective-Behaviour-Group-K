# Collective Behavior Project Group K — Replicating and Extending “Guidance by Multiple Sheepdogs Including Abnormalities”

## Team Members
| Name | GitHub Username |
|------|------------------|
| Adrien Doan | [@adrien-doan](https://github.com/adrien-doan) |
| Edita Džubur | [@editadzubur](https://github.com/editadzubur) |

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

###  Milestone 1 — First Report (Due **November 16, 2025**)

**Objectives**
- Conduct a literature review on collective behavior and shepherding algorithms.  
- Analyze the Tashiro et al. (2022) model and MSR algorithm in detail.  
- Define our implementation framework (Python or MATLAB) and assign group roles.  
- Implement the **baseline model** (flock + one dog, no anomalies).  

**Deliverables**
- Updated `README.md` with finalized plan.  
- Documented code for the basic sheep and single-dog model.  
- Short report outlining our methodology and equations.

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

This repository will serve as the main collaboration and documentation hub:
- Track progress and iterations through commits and branches.  
- Store all simulation code, experiment results, and reports.  
- Use GitHub milestones and issues to follow project goals.  
- Update this `README.md` regularly to reflect the current project status.

---

###  Keywords
`Collective Behavior` · `Swarm Robotics` · `Flocking` · `MSR Algorithm` · `Anomaly Tolerance` · `Agent-Based Modeling`

---

### 📚 Reference
Tashiro, M., Kubo, M., Sato, H., & Yamaguchi, A. (2022). *Guidance by multiple sheepdogs including abnormalities*. Artificial Life and Robotics, 27, 714–725.  
[https://doi.org/10.1007/s10015-022-00807-1](https://doi.org/10.1007/s10015-022-00807-1)
