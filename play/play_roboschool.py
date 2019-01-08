import gym
import roboschool

if __name__=='__main__':
    ENV='RoboschoolHumanoidFlagrunHarder-v1'
    env=gym.make(ENV)
    env.reset()

    for _ in range(10000):
        env.step(env.action_space.sample())
        env.render()