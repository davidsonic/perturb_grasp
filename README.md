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

### To run training experiments with human interaction:

1. Start service under play folder

```
python server.py
```

2. Start GUI interface under play folder

```
python gui.py
```

3. Start RL algorithms under reimplementation folder

 ```
 python run_mujoco_human.py --range==0.5
 ```


### To validate trained model with human interaction:

```
python python eval_ppo_server.py --range=0.6 --save_path='model_path' --seed=997
```


## Other Parameters

1. The server end defaultly uses 33000 port
2. Range denotes the strength of the force:

```
usage: eval_ppo_server.py [-h] [--env ENV] [--seed SEED]
                          [--save_path SAVE_PATH] [--range RANGE]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             environment ID (default: MyInvertedPendulum-v2)
  --seed SEED           RNG seed (default: 0)
  --save_path SAVE_PATH
                        save model path (default: /tmp/gym/ppo_1_inverted_pend
                        ulum_seeds/rand2-0/best_model)
  --range RANGE         adv force range (default: 0)
```


