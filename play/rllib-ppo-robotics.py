import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

ray.init()
config=ppo.DEFAULT_CONFIG.copy()

config['num_workers']=16
agent=ppo.PPOAgent(config, env='FetchPickAndPlace-v1')

for i in range(3000):
    result=agent.train()
    print(pretty_print(result))

    if i % 100==0:
        checkpoint=agent.save()
        print('checkpoint saved at ', checkpoint)