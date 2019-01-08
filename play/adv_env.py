import gym
import numpy as np
from mujoco_py import functions
import copy

def update_adversary(env, new_adv_max):
    from gym import spaces
    adv_max_force = new_adv_max
    adv_action_space = env.adv_action_space.shape[0]
    high_adv = np.ones(adv_action_space) * adv_max_force
    low_adv = -high_adv
    env.adv_action_space = spaces.Box(low_adv, high_adv)


if __name__=='__main__':
    ENV='FetchAdv-v1'
    env=gym.make(ENV)
    env.reset()
    update_adversary(env, 100)
    ac=env.sample_action()



    for i in range(10000):
        ac.pro=env.action_space.sample()
        ac.adv=env.adv_action_space.sample()
        if(i %20==0):
            i=0
            idx=env._adv_bindex
            nv = env.sim.model.nv
            qfrc_applied=np.zeros((nv),dtype=np.float64)
            force=np.array([0.1,0.2,0.3], dtype=np.float64)
            torque=np.array([0.3,0.2, 0.5], dtype=np.float64)
            point= np.array([0,0,0], dtype=np.float64)
            functions.mj_applyFT(env.sim.model, env.sim.data, force, torque, point ,32, qfrc_applied)
            env.sim.data.qfrc_applied[:]=qfrc_applied
            print('after',env.sim.data.qfrc_applied)


        env.step(ac)
        env.render()