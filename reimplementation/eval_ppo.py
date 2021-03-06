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

from queue import Queue

def policy_fn(name, ob_space, ac_space):
    return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                               hid_size=64, num_hid_layers=2)


def eval(env_id, seed, save_path, range):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    test_env=gym.make(env_id)
    test_env.seed(seed)


    ob_space=test_env.observation_space
    ac_space=test_env.action_space
    model = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    model.load(save_path)
    print('model loaded')
    # rew=PPO.test_hard(model, test_env, range)
    q= Queue()
    rew =PPO.test_random(model, test_env, True, range, q)

    test_env.close()
    return rew


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # environment must be correct
    parser.add_argument('--env', help='environment ID', default='MyInvertedPendulum-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=777)
    parser.add_argument('--save_path', default='/tmp/gym/ppo_1_inverted_pendulum_seeds/b-0/best_model', help='save model path')
    # parser.add_argument('--save_path', default='/tmp/gym/ppo_1_inverted_human/best_model', help='save model path')
    parser.add_argument('--range', default=0, type=float, help='adv force range')
    args = parser.parse_args()
    print('Testing params')
    print(args)
    reward= eval(args.env, args.seed, args.save_path, args.range)

    print(reward)

