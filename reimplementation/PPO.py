"""
Single agent framework using PPO as policy optimizer
"""

import time
from collections import deque
import numpy as np
import tensorflow as tf

from mpi4py import MPI
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
import os

from mujoco_py import functions
from gym import spaces

def traj_segment_generator(pi, env, horizon, stochastic):
    """
    Given policy and state, generate a trajectory of the agent
    :param pi:
    :param env:
    :param horizon:
    :param stochastic:
    :return:
    """
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()
    if isinstance(ob, dict):
        ob=ob['observation']

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        if isinstance(ob, dict):
            ob=ob['observation']
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob=env.reset()
            if isinstance(ob, dict):
                ob = ob['observation']
        t += 1



def traj_segment_generator_perturb(pi, env, horizon, stochastic, coeff, q):
    t=0
    action = env.sample_action()
    new = True
    ob = env.reset()
    if isinstance(ob, dict):
        ob = ob['observation']

    cur_ep_ret=0
    cur_ep_len=0
    ep_rets=[]
    ep_lens=[]

    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([action.pro for _ in range(horizon)])
    prevacs = acs.copy()

    # counter =0
    # adv_init = np.zeros_like(env.adv_action_space.sample(), dtype=np.float64)
    while True:
        prevac = action.pro
        action.pro , vpred = pi.act(stochastic, ob)
        action.adv = np.zeros_like(env.adv_action_space.sample(), dtype=np.float64)

        if t>0 and t%horizon ==0:
            yield {'ob': obs, 'rew': rews ,'vpred': vpreds ,'new': news,
                   'ac': acs, 'prevac': prevacs, 'nextvpred': vpred *(1-new),
                   'ep_rets': ep_rets, 'ep_lens': ep_lens}

            ep_rets=[]
            ep_lens=[]

        i = t % horizon
        obs[i]= ob
        vpreds[i]=vpred
        news[i]= new
        if not q.empty():
            adv_init = q.get() * coeff
            action.adv = adv_init
            print('action.adv: ', action.adv)
        acs[i]= action.pro
        prevacs[i]= prevac

        ob, rew, new, _ = env.step(action)
        # too slow, not working?
        # env.render()
        if isinstance(ob, dict):
            ob = ob['observation']
        rews[i]= rew

        cur_ep_ret += rew
        cur_ep_len += 1

        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret=0
            cur_ep_len=0
            ob=env.reset()
            if isinstance(ob, dict):
                ob = ob['observation']

        t+=1



def traj_segment_generator_random(pi, env, horizon, stochastic, coeff):
    t=0
    action = env.sample_action()
    new = True
    ob = env.reset()
    if isinstance(ob, dict):
        ob = ob['observation']

    cur_ep_ret=0
    cur_ep_len=0
    ep_rets=[]
    ep_lens=[]

    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([action.pro for _ in range(horizon)])
    prevacs = acs.copy()

    # counter =0
    # adv_init = np.zeros_like(env.adv_action_space.sample(), dtype=np.float64)
    cnt =0
    random_interval = 10
    while True:
        cnt+=1
        prevac = action.pro
        action.pro , vpred = pi.act(stochastic, ob)
        action.adv = np.zeros_like(env.adv_action_space.sample(), dtype=np.float64)

        if t>0 and t%horizon ==0:
            yield {'ob': obs, 'rew': rews ,'vpred': vpreds ,'new': news,
                   'ac': acs, 'prevac': prevacs, 'nextvpred': vpred *(1-new),
                   'ep_rets': ep_rets, 'ep_lens': ep_lens}

            ep_rets=[]
            ep_lens=[]

        i = t % horizon
        obs[i]= ob
        vpreds[i]=vpred
        news[i]= new
        if cnt % random_interval ==0:
            adv_init = env.adv_action_space.sample() * coeff
            action.adv = adv_init
            # print('action.adv: ', action.adv)
            cnt =0
        acs[i]= action.pro
        prevacs[i]= prevac

        ob, rew, new, _ = env.step(action)
        # too slow, not working?
        # env.render()
        if isinstance(ob, dict):
            ob = ob['observation']
        rews[i]= rew

        cur_ep_ret += rew
        cur_ep_len += 1

        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret=0
            cur_ep_len=0
            ob=env.reset()
            if isinstance(ob, dict):
                ob = ob['observation']

        t+=1




def evaluate(pi, env):
    ret = 0
    t = 1e3
    ob=env.reset()
    if isinstance(ob, dict):
        ob = ob['observation']
    done = False
    while not done and t > 0:
        ac = pi.act(False, ob)[0]
        ob, rew, done, _ = env.step(ac)
        if isinstance(ob, dict):
            ob=ob['observation']
        ret += rew
        t -= 1
    return ret


def test(pi, env):
    ret=0
    t=1e3
    ob=env.reset()
    if isinstance(ob, dict):
        ob=ob['observation']
    done=False
    while not done and t>0:
        ac=pi.act(False, ob)[0]
        env.render()
        ob, rew, done, _=env.step(ac)
        if isinstance(ob, dict):
            ob=ob['observation']
        ret+=rew
        t-=1
    return ret




# with perturb, time interval of force is fixed now
def test_random(pi, env, is_adv, range, q):
    print('Test with perturb.....')
    ret=0
    t=1e4
    # debug
    counter=0
    ob = env.reset()
    if isinstance(ob, dict):
        ob= ob['observation']
    done=False
    if is_adv:
        action=env.sample_action()
        adv_init = env.adv_action_space.sample() * range
    while not done and t>0:
        print('adv_init: ', adv_init)
        counter+=1
        ac=pi.act(False, ob)[0]
        if is_adv:
            action.pro=ac
            # random force ratio:
            if counter >= 100 and counter <=130:
                action.adv = adv_init
                if counter % 130==0:
                    counter =0
                    adv_init = env.adv_action_space.sample()*range
            else:
                action.adv=np.zeros_like(env.adv_action_space.sample(), dtype=np.float64)
            ob, rew, done, _=env.step(action)
            if isinstance(ob, dict):
                ob=ob['observation']
        else:
            ob, rew, done, _=env.step(ac)
            if isinstance(ob, dict):
                ob=ob['observation']
        env.render()
        ret+=rew
        t-=1

    return ret



def test_interactive(pi, env, is_adv, range, q):
    print('Test with perturb.....')
    ret=0
    t=1e4
    counter=0
    ob = env.reset()
    if isinstance(ob, dict):
        ob= ob['observation']
    done=False
    if is_adv:
        action=env.sample_action()
        adv_init = np.zeros_like(env.adv_action_space.sample(), dtype=np.float64)
    while not done and t>0:
        print('adv_init: ', adv_init)
        ac=pi.act(False, ob)[0]
        if is_adv:
            action.pro=ac
            if not q.empty():
                adv_init = q.get() * range
                action.adv = adv_init
            if any(adv_init):
                counter += 1
                if counter % 30 ==0:
                    counter =0
                    adv_init[:] = np.zeros_like(action.adv, dtype=np.float64)
            else:
                action.adv=np.zeros_like(env.adv_action_space.sample(), dtype=np.float64)
            ob, rew, done, _=env.step(action)
            if isinstance(ob, dict):
                ob=ob['observation']
        else:
            ob, rew, done, _=env.step(ac)
            if isinstance(ob, dict):
                ob=ob['observation']
        env.render()
        ret+=rew
        t-=1

    return ret



def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, test_env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        success_reward=10000,
        save_path='model/new_model', apply_force = False, coeff=1.0):
    # Setup losses and stuff
    # ----------------------------------------
    rew_mean = []
    if hasattr(env.observation_space, 'spaces'):
        ob_space = env.observation_space.spaces['observation']
    else:
        ob_space=env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    # if os.path.exists(os.path.join(save_path+'/checkpoint')):
    #     print('model loaded... ')
    #     pi.load(save_path+'/best_model')
    # else:
    #     print('train from scratch...')

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    if apply_force:
        seg_gen = traj_segment_generator_random(pi, env, timesteps_per_batch, stochastic=True, coeff=coeff)
    else:
        seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    test_interval=5
    test_start=0

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals(), save_path)
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values

        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult) 
                losses.append(newlosses)

        curr_rew = evaluate(pi, test_env)
        rew_mean.append(curr_rew)

        if not apply_force:
            print('evaluation reward: ', curr_rew)

        if test_start % test_interval ==0:
            if apply_force:
                test_rew=test_random(pi, test_env, True, range=coeff, q=None)
            else:
                test_rew= test(pi, test_env)
            print('test_reward',test_rew)



        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        episodes_so_far += len(lens)
        if len(lens) != 0:
            rew_mean.append(np.mean(rewbuffer))
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        test_start+=1

    return rew_mean


def learn_with_human(env, test_env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        success_reward=10000,
        save_path='model/new_model', data_queue=None):
    # Setup losses and stuff
    # ----------------------------------------
    rew_mean = []
    if hasattr(env.observation_space, 'spaces'):
        ob_space = env.observation_space.spaces['observation']
    else:
        ob_space=env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)


    U.initialize()
    adam.sync()

    # Prepare for rollouts
    seg_gen = traj_segment_generator_perturb(pi, env, timesteps_per_batch, stochastic=True, coeff=0.2, q=data_queue)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    test_interval=10
    eval_interval=5
    test_start=0
    eval_start=0

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)
        print('training part......')
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values

        print('optimize for %d epochs' % optim_epochs)
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)

        if eval_start % eval_interval ==0:
            print('evaluation part......')
            curr_rew = evaluate(pi, test_env)
            rew_mean.append(curr_rew)
            print('evalution reward: ', curr_rew)

        if test_start % test_interval ==0:
            print('testing part.......')
            test_rew=test_random(pi, test_env, True, 0.3, data_queue)
            print('test reward: ', test_rew)


        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        episodes_so_far += len(lens)
        if len(lens) != 0:
            rew_mean.append(np.mean(rewbuffer))
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        test_start+=1
        eval_start+=1

    return rew_mean


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]




