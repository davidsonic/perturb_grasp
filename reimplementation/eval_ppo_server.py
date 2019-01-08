import argparse
import logging
import os.path as osp

import MlpPolicy
import gym
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

import PPO
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from threading import Thread
from queue import Queue
from socket import AF_INET, socket, SOCK_STREAM
import numpy as np


def policy_fn(name, ob_space, ac_space):
    return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                               hid_size=64, num_hid_layers=2)




def eval(env_id, seed, save_path, range, q):
    print('main thread start....')
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    test_env=gym.make(env_id)
    test_env.seed(seed)

    ob_space=test_env.observation_space
    ac_space=test_env.action_space
    model = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    model.load(save_path)
    print('model loaded')
    rew =PPO.test_interactive(model, test_env, True, range, q)
    # rew = PPO.test_random(model, test_env, True, range, q)
    print('reward: ',rew)
    test_env.close()




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
                    # print('receive force_torque: ', force_torque)
                    q.put(np.array(force_torque))
                except:
                    pass
        except OSError:
            break




if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # environment must be correct
    parser.add_argument('--env', help='environment ID', default='MyInvertedPendulum-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--save_path', default='/tmp/gym/ppo_1_inverted_pendulum_seeds/rand2-0/best_model', help='save model path')
    parser.add_argument('--range', default=0, type=float, help='adv force range')
    args = parser.parse_args()

    # networking
    HOST= 'localhost'
    PORT = '33000'
    PORT = int(PORT)

    BUFSIZ=1024
    ADDR=(HOST, PORT)

    client_socket = socket(AF_INET, SOCK_STREAM)
    client_socket.connect(ADDR)
    client_socket.send(bytes('RL','utf8'))

    q = Queue()
    datathread = Thread(target=receive, args=(q,))
    mainthread = Thread(target=eval, args=(args.env, args.seed, args.save_path, args.range, q))
    datathread.start()
    mainthread.start()

    datathread.join()
    mainthread.join()

    client_socket.send(bytes('quit','utf8'))
