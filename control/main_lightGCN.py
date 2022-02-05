import copy
import glob
import os
import time
import types
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
import argparse
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
import networkx as nx
import scipy.sparse as sp
from utils import update_linear_schedule

from running_mean_std import RunningMeanStd

from pygcn.train import update_graph
from pygcn.models import GCN, lightGCN


def main(args):
    
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    
    eval_log_dir = args.log_dir + "_eval"
    
    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    
    torch.set_num_threads(1)
    device = torch.device(args.device)

    run_id = "alpha{}".format(args.gcn_alpha)
    if args.use_logger:
        from utils import Logger
        folder = "{}/{}".format(args.folder,run_id)
        logger = Logger(algo_name = args.algo, environment_name = args.env_name, folder = folder, seed=args.seed)
        logger.save_args(args)

        print ("---------------------------------------")
        print ('Saving to', logger.save_folder)
        print ("---------------------------------------")  

    else:
        print ("---------------------------------------")
        print ('NOTE : NOT SAVING RESULTS')
        print ("---------------------------------------") 
    all_rewards = []


    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space, args.env_name,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)


    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size,
                        actor_critic.base.output_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)



    ############################
    # GCN Model and optimizer
    if args.gcn_model == "gcn":
        gcn_model = GCN(nfeat=actor_critic.base.output_size, nhid=args.gcn_hidden)
    elif args.gcn_model == "light_gcn":
        gcn_model = lightGCN(nfeat=actor_critic.base.output_size, nhid=args.gcn_hidden)
    else:
        assert False, "arg gcn model invalid"
    gcn_model.to(device)
    gcn_optimizer = optim.Adam(gcn_model.parameters(),
                           lr=args.gcn_lr, weight_decay=args.gcn_weight_decay)
    gcn_loss = nn.NLLLoss()
    gcn_states = [[] for _ in range(args.num_processes)]
    Gs = [nx.Graph() for _ in range(args.num_processes)]
    node_ptrs = [ 0 for _ in range(args.num_processes)]
    rew_states = [ [] for _ in range(args.num_processes)]
    ############################

    episode_rewards = deque(maxlen=100)
    avg_fwdloss = deque(maxlen=100)
    rew_rms = RunningMeanStd(shape=())
    delay_rew = torch.zeros([args.num_processes,1])
    delay_step = torch.zeros([args.num_processes])

    start = time.time()
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob,\
                 recurrent_hidden_states, hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
            
            
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            delay_rew+=reward
            delay_step+=1

            for idx,(info,hid, eps_done) in enumerate(zip(infos,hidden_states,done)):

                if eps_done or delay_step[idx] == args.reward_freq:
                    reward[idx] = delay_rew[idx]
                    delay_rew[idx] = delay_step[idx] = 0
                else:
                    reward[idx] = 0


                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])


                if args.gcn_alpha < 1.0:
                    gcn_states[idx].append(hid)
                    node_ptrs[idx]+=1
                    if not eps_done:
                        Gs[idx].add_edge(node_ptrs[idx]-1,node_ptrs[idx])
                    if reward[idx] != 0. or eps_done:
                        rew_states[idx].append([node_ptrs[idx]-1,reward[idx]])
                    if eps_done:
                        adj = nx.adjacency_matrix(Gs[idx]) if len(Gs[idx].nodes)\
                                        else sp.csr_matrix(np.eye(1,dtype='int64'))
                        update_graph(gcn_model,gcn_optimizer,
                            torch.stack(gcn_states[idx]),adj,
                            rew_states[idx],gcn_loss,args,envs,device)
                        gcn_states[idx]=[]
                        Gs[idx]=nx.Graph()
                        node_ptrs[idx]=0
                        rew_states[idx] =[]


            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, hidden_states)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau, gcn_model, args.gcn_alpha, device)
        agent.update(rollouts)
        rollouts.after_update()


        ####################### Saving and book-keeping #######################
        if (j % int(num_updates/5.) == 0
                or j == num_updates - 1) and args.save_dir != "":
            print('Saving model')
            print()


            save_dir = "{}/{}/{}".format(args.save_dir,args.folder,run_id)
            save_path = os.path.join(save_dir, args.algo, 'seed' + str(args.seed)) + '_iter' + str(j)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            save_gcn = gcn_model
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
                save_gcn = copy.deepcopy(gcn_model).cpu()  

            save_model = [save_gcn, save_model,  hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + "ac.pt"))



        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {}\
             training episodes: mean/median reward {:.2f}/{:.2f},\
              min/max reward {:.2f}/{:.2f}, success rate {:.2f}, avg fwdloss {:.2f}\n".
                format(
                    j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards),
                    np.mean(avg_fwdloss),
                )
            )

            all_rewards.append(np.mean(episode_rewards))
            if args.use_logger:
                logger.save_task_results(all_rewards)
        ####################### Saving and book-keeping #######################

    envs.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--device', default='cuda:0',
                        help='CPU/GPU device to use')
    parser.add_argument('--gcn_model', default='gcn',
                        help='algorithm to use: gcn | light_gcn')
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--num-processes', type=int, default=8,
                        help='how many training CPU processes to use')
    parser.add_argument('--num-steps', type=int, default=128,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=4,
                        help='number of batches for ppo')
    parser.add_argument('--clip-param', type=float, default=0.1,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=int(10e6),
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='ZaxxonNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='./log',
                        help='directory to save agent logs (default: ./log)')
    parser.add_argument('--save-dir', default='./save_dir',
                        help='directory to save agent logs (default: ./save_dir)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use_linear_lr_decay', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')

    parser.add_argument('--use-logger', action='store_true', default=True,
                        help='log the results') 
    parser.add_argument('--folder', default='./results',
                        help='folder to save results')
    parser.add_argument('--gcn_epochs', type=int, default=1,
                        help='Number of epochs to train.')
    parser.add_argument('--gcn_lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--gcn_weight_decay', type=float, default=0e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--gcn_hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--gcn_alpha', type=float, default=0.9,
                        help='Mixing coefficient between TD returns and GCN returns.')
    parser.add_argument('--gcn_lambda', type=float, default=10.,
                        help='Mixing coefficient between GCN losses.')
    parser.add_argument('--reward_freq', type=int, default=1,
                        help='Reward frequency.')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)

