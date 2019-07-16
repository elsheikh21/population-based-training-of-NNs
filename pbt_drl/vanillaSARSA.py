import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp 
import operator
from networks import ReplayBuffer, DeepDoubleSarsa
import gym
import random
from collections import Counter



epsilon_start, epsilon_stop, epsilon_decay = 1.0, 0.1, 1200
EPOCH = 20
EPISODES = 2000
BATCH_SIZE = 128
BUFFER_SIZE = 10000
target_update = 40


device = "cuda" if torch.cuda.is_available() else "cpu"


def gym_act(env, q1, q2, epsilon):
    if np.random.rand(1)[0] < epsilon:
        return env.action_space.sample()
    else:
        avg = [i+j for i,j in zip(q1 ,q2)]
        return np.argmax(np.array(avg)) 

def gymEvaluate(env, model1, model2, numEpisodes=20):
    obs_len = env.observation_space.shape[0]
    test_eps = 0.0
    rews = []
    for episode in range(1, numEpisodes+1):
        env.seed(random.randint(0, 999999)) #* For pure randomness of the environment instance
        done = False
        total_reward = 0

        obs1 = env.reset()
        obs = Variable(torch.from_numpy(obs1))
        obs = obs.view(-1, obs_len)
        obs = obs.float()
        obs = obs.to(device)
        qa = model1(obs)
        qb = model2(obs)
        qa = np.squeeze(qa.cpu().data.numpy())
        qb = np.squeeze(qb.cpu().data.numpy())
        a = gym_act(env, qa, qb, test_eps)
        t = 0

        while not done:
                env.render()
                n_obs1, r, done, _ = env.step(a)
                total_reward +=r
                n_obs = Variable(torch.from_numpy(n_obs1))
                n_obs = n_obs.view(-1, obs_len)
                n_obs = n_obs.float()
                n_obs = n_obs.to(device)
                n_qa = model1(n_obs)
                n_qb = model2(n_obs)
                n_qa = np.squeeze(n_qa.cpu().data.numpy())
                n_qb = np.squeeze(n_qb.cpu().data.numpy())
                an = gym_act(env, n_qa, n_qb, test_eps) 
                a = an
                t += 1
        rews.append(total_reward)
    evalAvg = np.mean(rews)
    evalStd = np.std(rews)
    evalMax = max(rews)
    print("Evaluation  total reward for {} episodes => max: {}, mean: {}, std: {}.".format(numEpisodes, evalMax, evalAvg, evalStd))
    return evalAvg


def lunarLanderTrain():
    env = gym.make("LunarLander-v2")
    env.seed(random.randint(0, 999999))
    obs_len = env.observation_space.shape[0]
    model1 = DeepDoubleSarsa(8, 4, True, lr=0.00025)
    model2 = DeepDoubleSarsa(8, 4, True, lr=0.00025)
    model1.to(device)
    model2.to(device)

    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    gamma = 0.99
    epsilon = epsilon_start
    max_score = 0
    returns = []
    evals = []
    for e in range(1, EPISODES+1):

        done = False
        total_reward = 0

        obs1 = env.reset()
        obs = Variable(torch.from_numpy(obs1))
        obs = obs.view(-1, obs_len)
        obs = obs.float()
        obs = obs.to(device)
        qa = model1(obs)
        qb = model2(obs)
        qa = np.squeeze(qa.cpu().data.numpy())
        qb = np.squeeze(qb.cpu().data.numpy())
        a = gym_act(env, qa, qb, epsilon)
        t = 0

        while not done:
            n_obs1, r, done, _ = env.step(a)
            total_reward +=r
            n_obs = Variable(torch.from_numpy(n_obs1))
            n_obs = n_obs.view(-1, obs_len)
            n_obs = n_obs.float()
            n_obs = n_obs.to(device)
            n_qa = model1(n_obs)
            n_qb = model2(n_obs)
            n_qa = np.squeeze(n_qa.cpu().data.numpy())
            n_qb = np.squeeze(n_qb.cpu().data.numpy())
            an = gym_act(env, n_qa, n_qb, epsilon) 
            if done:
                replay_buffer.push(obs1, a, r, n_obs1, an, 1.0, n_qa, n_qb)
            else:
                replay_buffer.push(obs1, a, r, n_obs1, an, 0.0, n_qa, n_qb)
            a = an
            if len(replay_buffer)>=BATCH_SIZE:
                if e%target_update:
                    s, ac, r, sp, ap, d, q1nn, q2nn = replay_buffer.sample(BATCH_SIZE)
                    loss = model1.update([s, ac, r, sp, ap, d], q2nn, gamma)
                else:
                    model1.save('target')
                    model2.load('target')

            obs1 = n_obs1
            t += 1
        returns.append(total_reward)
        print("Episode: {} Return: {}".format(e, total_reward))
        if epsilon>epsilon_stop:
            epsilon -= (epsilon_start - epsilon_stop)/epsilon_decay

        if max_score < total_reward:
            max_score = total_reward
    
        if not e%40:
            print("Episode {} Max score: {}     Eps: {}".format(e, max_score, epsilon))
            avg = gymEvaluate(env, model1, model2)
            evals.append(avg)
    avg = gymEvaluate(env, model1, model1)
    evals.append(avg)
    np.save("lunarTrainingRewards01", returns)
    np.save("lunarEvaluationRewards01", avg)

if __name__ == "__main__":
    lunarLanderTrain()