"""
Training script for human-agent framework
"""

import argparse
import logging
import os.path as osp

import MlpPolicy
import gym
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.results_plotter import ts2xy
from baselines.common.plot_util import load_results
from baselines.bench import Monitor
import numpy as np
from pandas import read_csv

import PPO
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from threading import Thread
from queue import Queue
from socket import AF_INET, socket, SOCK_STREAM


def policy_fn(name, ob_space, ac_space):
    return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                               hid_size=64, num_hid_layers=2)

# load dir
log_dir='/tmp/gym/ppo_1_inverted_pendulum'
os.makedirs(log_dir, exist_ok=True)
cur_best, n_steps = -np.inf, 0

def plot_callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, cur_best
    # Print stats every 1000 calls
    if (n_steps + 1) % 2 == 0:
        # Evaluate policy performance
        # x, y = ts2xy(load_results(log_dir), 'timesteps')
        # df=load_results(log_dir)
        df=read_csv(os.path.join(log_dir, 'monitor.csv'),index_col=None, comment='#')
        x=np.cumsum(df.l.values)
        y=df.r.values
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(cur_best, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > cur_best:
                cur_best = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['pi'].save(log_dir + '/best_model')
    n_steps += 1
    return False



def receive(q):
    print('daemon thread start....')
    global client_socket
    global BUFSIZ
    while True:
        try:
            msg = client_socket.recv(BUFSIZ).decode('utf8')
            if msg!='':
                try:
                    msg = msg.split(',')
                    force_torque = list(map(float, msg))
                    q.put(np.array(force_torque))
                except:
                    pass
        except OSError:
            break




def train(env_id, num_iters, seed, success_reward, save_path, q):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    env=Monitor(env, log_dir, allow_early_resets=True)
    test_env = gym.make(env_id)

    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    test_env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    rew = PPO.learn_with_human(env, test_env, policy_fn,
                    max_iters=num_iters,
                    timesteps_per_batch=2048,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                    gamma=0.99, lam=0.95, schedule='constant', success_reward=success_reward,
                    save_path=save_path, callback=plot_callback, data_queue=q,
                    )
    env.close()
    return rew


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='MyInvertedPendulum-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=10)
    parser.add_argument('--sr', default=1000.0, help='success reward')
    args = parser.parse_args()
    print('Training params')
    print(args)

    HOST = 'localhost'
    PORT ='33000'
    PORT = int(PORT)

    BUFSIZ = 1024
    ADDR = (HOST, PORT)

    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.connect(ADDR)
    client_socket.send(bytes('RL', 'utf8'))

    q = Queue()
    datathread = Thread(target=receive, args=(q, ))
    mainthread = Thread(target=train, args=(args.env, 500, args.seed, args.sr, log_dir, q))

    datathread.start()
    mainthread.start()

    datathread.join()
    mainthread.join()

    client_socket.send(bytes('quit', 'utf8'))