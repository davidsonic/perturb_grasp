import gym
import numpy as np

if __name__=='__main__':
    ENV='HumanoidStandup-v2'
    env=gym.make(ENV)
    env.reset()
    # ac=env.sample_action()
    print(env.action_space)
    ac=env.action_space.sample()
    print(ac)
    print(env.action_space.low)
    for i in range(10000):
        env.step(ac)
        env.render()
        ac = env.action_space.sample()
        # if(i%20==0):
        #     idx=env._adv_bindx
        #     print('xfrc_applied', env.sim.data.xfrc_applied[idx])
