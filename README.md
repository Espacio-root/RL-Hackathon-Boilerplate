# 🐕 Unitree Go2 Locomotion Hackathon: Participant Guide

Welcome to the **RL Locomotion Hackathon**! Your goal is to train a Reinforcement Learning policy capable of controlling a Unitree Go2 quadruped robot. You will submit an agent that will be evaluated on our backend for stability, velocity tracking, and robustness.

## 📚 Table of Contents
1. [The Challenge](#the-challenge)
2. [Environment Setup](#environment-setup)
3. [Training Your Policy](#training-your-policy)
4. [Submission Structure](#submission-structure)
5. [The Agent Interface](#the-agent-interface)
6. [Evaluation Criteria](#evaluation-criteria)

---

## The Challenge

You must train a policy to control the 12 joints of a Go2 robot to achieve a target forward velocity of **0.5 m/s - 1.0 m/s**.

* **Observation Space:** 45 dimensions (Linear/Angular velocities, DOF positions, DOF velocities).
* **Action Space:** 12 dimensions (Joint position targets).
* **Control Frequency:** 50Hz (Simulation step: 0.02s).

The backend will run your agent in a `Genesis` simulation environment.

---

## Environment Setup

You need to install the Genesis simulator and the RSL-RL library.

### 1. Requirements
Ensure you are using **Python 3.8+** and have a GPU available.

### 2. Installation
The training script enforces specific versions. Install the following:

```bash
# Install Genesis (ensure you follow specific cuda instructions for your machine)
pip install genesis-world

# Install RSL-RL (Strict version requirement)
pip install rsl-rl-lib==2.2.4
```

---

## Training Your Policy

We have provided a baseline training script: `go2_train.py`.

### Running the Training
To start training with the PPO algorithm:

```bash
python go2_train.py --exp_name my_submission --max_iterations 501
```

### Output
After training, the script will generate a folder `logs/my_submission/` containing:
* `model_X.pt`: The model checkpoints.
* `cfgs.pkl`: A pickle file containing the environment and model configurations used during training. **(Important: You need this for submission)**.

### Modifying the Training
You are free to modify the `go2_train.py` hyperparameters (learning rate, PPO clip range, network architecture) to improve performance. 

> **⚠️ WARNING:** Do not change the **Observation Space (45)** or **Action Space (12)** dimensions. The evaluation backend will fail if these dimensions do not match.

---

## Submission Structure

Your submission must be a zipped folder containing a specific file structure. The entry point for our backend is `agent.py`.

### Required Directory Structure

```text
submission_folder/
├── agent.py                 # MANDATORY: The interface class
├── requirements.txt         # OPTIONAL: Extra dependencies
└── checkpoints/             # RECOMMENDED: Your trained models
    ├── model_1500.pt        # Your best model weights
    └── cfgs.pkl             # Your training config (needed to rebuild net)
```

### `requirements.txt`
If you use libraries other than `torch`, `numpy`, `rsl_rl`, and `pickle`, list them here.

---

## The Agent Interface

You must provide an `agent.py` file containing a class named `Agent`. The backend will import this class, initialize it once, and call `apply()` at every simulation step.

### The `Agent` Class Specification

#### 1. Initialization (`__init__`)
This method should load your model weights and prepare the network.
* **Path Handling:** Use `os.getcwd()` and `os.path.join` to locate your checkpoints relative to the root of your submission. **Do not use absolute paths.**
* **Architecture Reconstruction:** The provided template uses `cfgs.pkl` to read the network architecture (hidden dims, activation functions) so it matches what you trained.

#### 2. The `apply(self, obs)` method
* **Input:** `obs` (torch.Tensor) of shape `(num_envs, 45)`.
* **Output:** `actions` (torch.Tensor) of shape `(num_envs, 12)`.
* **Context:** This function is called every control step. Ensure it runs efficiently (avoid heavy I/O here).

---

## Evaluation Criteria

Your agent will be graded on a hidden test set of scenarios. The primary metric is the **Reward Function** defined in `go2_train.py`, which prioritizes:

1.  **Velocity Tracking:** Maintaining $v_x \approx 0.6 m/s$.
2.  **Stability:** Minimizing vertical velocity ($v_z$) and body orientation errors (pitch/roll).
3.  **Efficiency:** Minimizing action rate (jitter) and torque usage.

### Grading Logic Flow

1.  **Load:** Backend initializes `Agent()`.
2.  **Loop:** For `T` timesteps:
    * Backend sends observation $O_t$.
    * Agent returns action $A_t$.
    * Physics engine steps forward.
    * Reward $R_t$ is accumulated.
3.  **Score:** Final average reward per episode.
