import argparse
import logging
import os.path as osp

import MlpPolicy
import gym
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

import PPO_RARL
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

def policy_fn(name, ob_space, ac_space):
    return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                               hid_size=64, num_hid_layers=2)


def eval(env_id, seed, save_pro, save_adv):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    test_env=gym.make(env_id)
    test_env.seed(seed)


    ob_space=test_env.observation_space
    ac_space=test_env.action_space
    pro_model = policy_fn("pro_pi", ob_space, ac_space)  # Construct network for new policy
    pro_model.load(save_pro)
    print('pro model loaded')


    adv_space = test_env.adv_action_space
    adv_model = policy_fn("adv_pi", ob_space, adv_space)
    adv_model.load(save_adv)
    print('adv model loaded')

    rew=PPO_RARL.test_hard(model, test_env, range)

    test_env.close()
    return rew


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # environment must be correct
    parser.add_argument('--env', help='environment ID', default='MyInvertedPendulum-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--save_pro', default='/tmp/gym/rarl_2/pro_best_model', help='save pro model path')
    parser.add_argument('--save_adv', default='/tmp/gym/rarl_2/adv_best_model', help='save adv model path')
    args = parser.parse_args()
    print('Testing params')
    print(args)
    reward= eval(args.env, args.seed, args.save_pro, args.save_adv)

    print(reward)

