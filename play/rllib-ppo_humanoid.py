import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

ray.init()
config=ppo.DEFAULT_CONFIG.copy()
config['num_gpus']=4
config['num_workers']=8
agent=ppo.PPOAgent(config=config, env='Humanoid-v2')

agent.restore('/home/jiali/ray_results/PPO_Humanoid-v2_2018-12-10_15-42-32z6xg115i/checkpoint-901')

for i in range(1000):
    result=agent.train()
    print(pretty_print(result))

    if i %100==0:
        checkpoint=agent.save()
        print('checkpoint saved at', checkpoint)