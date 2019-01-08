# RL Grasping

Unlike existing repositories, the code is based on latest mujoco_py API and gym environment with enhancing features.
The repository contains the following functions:

### Human Interaction Interface

- Server end responsible for dispatching messages from clients

- GUI client supporting human interaction (perturbation)

- RL client responsible for algorithm training (RL thread + daemon thread)

The interactive training process is asynchronous and human interaction with RL agent is facilitated with the use of multithreading Queue.

### RL algorithm

- Original Training/Testing pipeline of PPO implementation

- Multiagent Training/Testing PPO pipeline

- Random perturbation training/testing PPO pipeline

- Interactive Training/Testing pipeline


### Visualization

- Training curve plots with multiple seeds


## Installation

1. Install Mujoco [MuJoCo website](https://www.roboti.us/license.html)
2. Install self_brewed_mujoco_py at [Self mujoco-py](https://github.com/davidsonic/self_brewed_mujoco_py)
3. Install self_brewed_gym at [Self gym](https://github.com/davidsonic/self_brewed_gym)
4. Install this code repository.


## Usage


