"""
RARL Framework
"""


import time

from collections import deque
import tensorflow as tf, numpy as np
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from utilities import timeSince
import os

from mujoco_py import functions



def traj_segment_generator(pro_pi, adv_pi, env, horizon, stochastic):
    t = 0
    ac = env.sample_action()  # not used, just so we have the datatype
    pro_ac = ac.pro
    adv_ac = ac.adv

    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()
    if isinstance(ob, dict):
        ob = ob['observation']

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    pro_vpreds = np.zeros(horizon, 'float32')
    adv_vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    pro_acs = np.array([pro_ac for _ in range(horizon)])
    adv_acs = np.array([adv_ac for _ in range(horizon)])
    pro_prevacs = pro_acs.copy()
    adv_prevacs = adv_acs.copy()

    while True:
        pro_prevac = pro_ac
        adv_prevac = adv_ac
        pro_ac, pro_vpred = pro_pi.act(stochastic, ob)
        adv_ac, adv_vpred = adv_pi.act(stochastic, ob)
        ac.pro = pro_ac
        ac.adv = adv_ac
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value

        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "pro_vpred": pro_vpreds, "adv_vpred": adv_vpreds, "new": news,
                    "pro_ac": pro_acs, "adv_ac": adv_acs, "pro_prevac": pro_prevacs, "adv_prevac": adv_prevacs,
                    "pro_nextvpred": pro_vpred * (1 - new), "adv_nextvpred": adv_vpred * (1 - new),
                    "ep_rets": ep_rets, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        pro_vpreds[i] = pro_vpred
        adv_vpreds[i] = adv_vpred
        news[i] = new
        pro_acs[i] = pro_ac
        adv_acs[i] = adv_ac
        pro_prevacs[i] = pro_prevac
        adv_prevacs[i] = adv_prevac

        ob, rew, new, _ = env.step(ac)
        # new environ
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
            ob = env.reset()
            if isinstance(ob, dict):
                ob = ob['observation']
        t += 1


# without perturb
def evaluate(pi, env, rollouts=10):
    print('Evaluate without perturb.....')
    retlst = []
    for _ in range(rollouts):
        ret = 0
        t = 1e3
        ob = env.reset()
        if isinstance(ob, dict):
            ob = ob['observation']
        done = False
        while not done and t > 0:
            ac = pi.act(False, ob)[0]
            ob, rew, done, _ = env.step(ac)
            if isinstance(ob, dict):
                ob = ob['observation']
            ret += rew
            t -= 1
        retlst.append(ret)
    return np.mean(retlst)


# with perturb
def test(pi, env, is_adv):
    print('Test with perturb.....')
    ret=0
    t=1e3
    ob=env.reset()
    if isinstance(ob, dict):
        ob= ob['observation']
    done=False
    if is_adv:
        action=env.sample_action()
    while not done and t>0:
        ac=pi.act(False, ob)[0]
        if is_adv:
            action.pro=ac
            # random force
            action.adv=env.adv_action_space.sample()
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


# debug
def genRandomForce(env, range):
    idx = env._adv_bindex
    # print('idx', idx)
    nv = env.sim.model.nv
    qfrc_applied = np.zeros((nv), dtype=np.float64)
    force= np.random.rand(3).astype(np.float64) * range
    torque = np.random.rand(3).astype(np.float64) * range
    point = np.array([0,0,0], dtype=np.float64)
    functions.mj_applyFT(env.sim.model, env.sim.data, force, torque, point, idx, qfrc_applied)
    env.sim.data.qfrc_applied[:]=qfrc_applied
    print('action.adv', qfrc_applied)



def test_hard(pi, env, range):
    ret =0
    t=1e3
    ob = env.reset()
    done = False
    while not done and t>0:
        ac= pi.act(False, ob)[0]
        env.render()
        if t % 20 ==0:
            genRandomForce(env, range)
        ob, rew, done, _ = env.step(ac)
        ret+= rew
        t-=1
    return ret



def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    pro_vpred = np.append(seg["pro_vpred"], seg["pro_nextvpred"])
    adv_vpred = np.append(seg["adv_vpred"], seg["adv_nextvpred"])
    T = len(seg["rew"])
    seg["pro_adv"] = pro_gaelam = np.empty(T, 'float32')
    seg["adv_adv"] = adv_gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    pro_lastgaelam = 0
    adv_lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        pro_delta = rew[t] + gamma * pro_vpred[t+1] * nonterminal - pro_vpred[t]
        adv_delta = -rew[t] - gamma * pro_vpred[t+1] * nonterminal + pro_vpred[t]
        pro_gaelam[t] = pro_lastgaelam = pro_delta + gamma * lam * nonterminal * pro_lastgaelam
        adv_gaelam[t] = adv_lastgaelam = adv_delta + gamma * lam * nonterminal * adv_lastgaelam
    seg["pro_tdlamret"] = seg["pro_adv"] + seg["pro_vpred"]
    seg["adv_tdlamret"] = seg["adv_adv"] + seg["adv_vpred"]


def learn(env, test_env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        success_reward=1000,
        save_path='model/new_model'):
    # Setup losses and stuff
    # ----------------------------------------
    rew_mean = []

    # ob_space = env.observation_space
    if hasattr(env.observation_space, 'spaces'):
        ob_space = env.observation_space.spaces['observation']
    else:
        ob_space = env.observation_space
    pro_ac_space = env.action_space
    adv_ac_space = env.adv_action_space

    pro_pi = policy_func("pro_pi", ob_space, pro_ac_space) # Construct network for new policy
    pro_oldpi = policy_func("pro_oldpi", ob_space, pro_ac_space) # Network for old policy
    adv_pi = policy_func("adv_pi", ob_space, adv_ac_space) # Construct network for new adv policy
    adv_oldpi = policy_func("adv_oldpi", ob_space, adv_ac_space) # Network for old adv policy

    pro_atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    adv_atarg = tf.placeholder(dtype=tf.float32, shape=[None]) 
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    pro_ac = pro_pi.pdtype.sample_placeholder([None])
    adv_ac = adv_pi.pdtype.sample_placeholder([None])

    pro_kloldnew = pro_oldpi.pd.kl(pro_pi.pd) # compute kl difference
    adv_kloldnew = adv_oldpi.pd.kl(adv_pi.pd)
    pro_ent = pro_pi.pd.entropy()
    adv_ent = adv_pi.pd.entropy()
    pro_meankl = tf.reduce_mean(pro_kloldnew)
    adv_meankl = tf.reduce_mean(adv_kloldnew)
    pro_meanent = tf.reduce_mean(pro_ent)
    adv_meanent = tf.reduce_mean(adv_ent)
    pro_pol_entpen = (-entcoeff) * pro_meanent
    adv_pol_entpen = (-entcoeff) * adv_meanent

    pro_ratio = tf.exp(pro_pi.pd.logp(pro_ac) - pro_oldpi.pd.logp(pro_ac)) # pnew / pold
    adv_ratio = tf.exp(adv_pi.pd.logp(adv_ac) - adv_oldpi.pd.logp(adv_ac)) 
    pro_surr1 = pro_ratio * pro_atarg # surrogate from conservative policy iteration
    adv_surr1 = adv_ratio * adv_atarg 
    pro_surr2 = tf.clip_by_value(pro_ratio, 1.0 - clip_param, 1.0 + clip_param) * pro_atarg #
    adv_surr2 = tf.clip_by_value(adv_ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_atarg
    pro_pol_surr = - tf.reduce_mean(tf.minimum(pro_surr1, pro_surr2)) # PPO's pessimistic surrogate (L^CLIP)
    adv_pol_surr = - tf.reduce_mean(tf.minimum(adv_surr1, adv_surr2))
    pro_vf_loss = tf.reduce_mean(tf.square(pro_pi.vpred - ret))
    adv_vf_loss = tf.reduce_mean(tf.square(adv_pi.vpred - ret))
    pro_total_loss = pro_pol_surr + pro_pol_entpen + pro_vf_loss
    adv_total_loss = adv_pol_surr + adv_pol_entpen + adv_vf_loss
    pro_losses = [pro_pol_surr, pro_pol_entpen, pro_vf_loss, pro_meankl, pro_meanent]
    pro_loss_names = ["pro_pol_surr", "pro_pol_entpen", "pro_vf_loss", "pro_kl", "pro_ent"]
    adv_losses = [adv_pol_surr, adv_pol_entpen, adv_vf_loss, adv_meankl, adv_meanent]
    adv_loss_names = ["adv_pol_surr", "adv_pol_entpen", "adv_vf_loss", "adv_kl", "adv_ent"]

    pro_var_list = pro_pi.get_trainable_variables()
    adv_var_list = adv_pi.get_trainable_variables()
    pro_lossandgrad = U.function([ob, pro_ac, pro_atarg, ret, lrmult], pro_losses + [U.flatgrad(pro_total_loss, pro_var_list)])
    adv_lossandgrad = U.function([ob, adv_ac, adv_atarg, ret, lrmult], adv_losses + [U.flatgrad(adv_total_loss, adv_var_list)])
    pro_adam = MpiAdam(pro_var_list, epsilon=adam_epsilon)
    adv_adam = MpiAdam(adv_var_list, epsilon=adam_epsilon)

    pro_assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(pro_oldpi.get_variables(), pro_pi.get_variables())])
    adv_assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(adv_oldpi.get_variables(), adv_pi.get_variables())])
    pro_compute_losses = U.function([ob, pro_ac, pro_atarg, ret, lrmult], pro_losses)
    adv_compute_losses = U.function([ob, adv_ac, adv_atarg, ret, lrmult], adv_losses)

    # if os.path.exists(os.path.join(save_path+'/checkpoint')):
    #     print('model loaded... ')
    #     pro_pi.load(save_path+'/pro_best_model')
    #     adv_pi.load(save_path+'/adv_best_model')
    # else:
    #     print('train from scratch...')

    U.initialize()

    pro_adam.sync()
    adv_adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pro_pi, adv_pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    test_inverval=2
    test_start=0

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


        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, pro_ac, adv_ac, pro_atarg, adv_atarg, pro_tdlamret, adv_tdlamret = seg["ob"], seg["pro_ac"], seg["adv_ac"], seg["pro_adv"], seg["adv_adv"], seg["pro_tdlamret"], seg["adv_tdlamret"]
        pro_vpredbefore = seg["pro_vpred"] # predicted value function before udpate
        adv_vpredbefore = seg["adv_vpred"]
        pro_atarg = (pro_atarg - pro_atarg.mean()) / pro_atarg.std() # standardized advantage function estimate
        adv_atarg = (adv_atarg - adv_atarg.mean()) / adv_atarg.std()

        # For pro policy
        d = Dataset(dict(ob=ob, ac=pro_ac, atarg=pro_atarg, vtarg=pro_tdlamret), shuffle=not pro_pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pro_pi, "ob_rms"): pro_pi.ob_rms.update(ob) # update running mean/std for policy

        pro_assign_old_eq_new() # set old parameter values to new parameter values

        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            pro_losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = pro_lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                pro_adam.update(g, optim_stepsize * cur_lrmult) 
                pro_losses.append(newlosses)

        pro_losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = pro_compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            pro_losses.append(newlosses)            
        pro_meanlosses,_,_ = mpi_moments(pro_losses, axis=0)


        # For adv policy
        d = Dataset(dict(ob=ob, ac=adv_ac, atarg=adv_atarg, vtarg=adv_tdlamret), shuffle=not adv_pi.recurrent)
        if hasattr(adv_pi, "ob_rms"): adv_pi.ob_rms.update(ob)
        adv_assign_old_eq_new()

        for _ in range(optim_epochs):
            adv_losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = adv_lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adv_adam.update(g, optim_stepsize * cur_lrmult) 
                adv_losses.append(newlosses)

        adv_losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = adv_compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            adv_losses.append(newlosses)            
        adv_meanlosses,_,_ = mpi_moments(adv_losses, axis=0)


        curr_rew = evaluate(pro_pi, test_env)
        rew_mean.append(curr_rew)
        print(curr_rew)
        if test_start % test_inverval==0:
            test_rew=test(pro_pi, test_env, True)
            print('test_reward', test_rew)


        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        test_start+=1

        print(('%s (%d %d%%) %d' % (
            timeSince(tstart, float(iters_so_far) / max_iters),
            iters_so_far, float(iters_so_far) / max_iters * 100, curr_rew)))

    return np.array(rew_mean)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
