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
import time



epsilon_start, epsilon_stop, epsilon_decay = 1.0, 0.1, 1200
EPOCH = 20
EPISODES = 100
BATCH_SIZE = 64
BUFFER_SIZE = 1000
target_update = 51
BEST = 4
WORST = 4

ENV = 'LunarLander-v2'


device = "cuda" if torch.cuda.is_available() else "cpu"

def exploitPopulation(train_acc, best=1, worst=1):
    #* best parameters takes how many bestPerformers are sought
    #? EXPLOIT
    k = Counter(train_acc)
    ordered = k.most_common()
    assert len(train_acc) >= best+worst, "Best and worst performers will overlap. Check your numbers!"
    bestPerformers = [i[0] for i in ordered[:best]]
    assert worst >= 1 , "No number of worst performers specified. This will result in regular parallel training."
    worstPerformers = []
    if worst >= 1:
        worstPerformers = [i[0] for i in ordered[-worst:]]

    return bestPerformers, worstPerformers

def exploreIndividual(workerID, model1, model2, bestPerformers, worstPerformers, train_acc):
    #? EXPLORE
    env = gym.make(ENV)
    env.seed(random.randint(0, 999999))   

    bestPerformer = random.choice(bestPerformers)
    
    if workerID in worstPerformers:
        model1.load('models/sarsa/modelA{}'.format(bestPerformer)) 
        model2.load('models/sarsa/modelB{}'.format(bestPerformer))
        print("Model {} is changed to Model {}".format(workerID, bestPerformer))

        #? Perturbation sequence
        seed = np.random.randint(0, 1000, 1)[0]
        model1.perturb(workerID, seed)
        model2.perturb(workerID, seed)

        model1.save('models/sarsa/modelA{}'.format(workerID)) #* Save method saves both weights and optimizer hyperparameters
        model2.save('models/sarsa/modelB{}'.format(workerID)) #* Save method saves both weights and optimizer hyperparameters

        train_acc[workerID] = gymEvaluate(workerID, None, model1, model2)

def gym_act(env, q1, q2, epsilon):
    if np.random.rand(1)[0] < epsilon:
        return env.action_space.sample()
    else:
        avg = [i+j for i,j in zip(q1 ,q2)]
        return np.argmax(np.array(avg))  

def gymEvaluate(workerID, epoch, model1, model2, numEpisodes=20):
    env = gym.make(ENV)
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
                # env.render()
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
                obs1 = n_obs1
                t += 1
        rews.append(total_reward)
    evalAvg = np.mean(rews)
    evalStd = np.std(rews)
    evalMax = max(rews)
    if epoch:
        np.save("evalRewsM{}E{}".format(workerID, epoch), np.array(rews))
    print("Evaluation  Model{}, total reward for {} episodes => max: {}, mean: {}, std: {}.".format(workerID, numEpisodes, evalMax, evalAvg, evalStd))
    return evalAvg

def pbtSarsa(workerID, model1, model2, device, eps, epoch, train_return, target=False):
    act_len = 4
    env = gym.make(ENV)
    env.seed(random.randint(0, 999999))
    obs_len = env.observation_space.shape[0]

    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    gamma = 0.99
    epsilon = eps
    max_score = 0
    total_rews = []
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
                replay_buffer.push(obs1, a, r, n_obs1, an, 1.0, np.squeeze(n_qa.cpu().data.numpy()), np.squeeze(n_qb.cpu().data.numpy()))
            else:
                replay_buffer.push(obs1, a, r, n_obs1, an, 0.0, np.squeeze(n_qa.cpu().data.numpy()), np.squeeze(n_qb.cpu().data.numpy()))
            a = an
            if len(replay_buffer)>=BATCH_SIZE:
                if e%target_update:
                    s, ac, r, sp, ap, d, q1nn, q2nn = replay_buffer.sample(BATCH_SIZE)
                    loss = model1.update([s, ac, r, sp, ap, d], q2nn, gamma)
                else:
                    model1.save('target{}'.format(workerID))
                    model2.load('target{}'.format(workerID))

            obs1 = n_obs1
            t += 1

        total_rews.append(total_reward)
        if epsilon>epsilon_stop:
            epsilon -= (epsilon_start - epsilon_stop)/epsilon_decay

        if max_score < total_reward:
            max_score = total_reward


    model1.save('models/sarsa/modelA{}'.format(workerID))
    model2.save('models/sarsa/modelB{}'.format(workerID))
    np.save("total_rews_model{}_epoch{}".format(workerID, epoch), total_rews)
    print("Model {} training max score: {}".format(workerID, max_score))
    avgEvalScore = gymEvaluate(workerID, epoch, model1, model2, numEpisodes=20)
    print("Model {} evaluation average score: {}".format(workerID, avgEvalScore))
    train_return[workerID] = avgEvalScore


def pbtRun():
    act_len = 4
    eps = 1.0
    lrs = list(np.arange(0.001, 0.1, 0.005)) # 20 individuals
    num_processes = len(lrs)
    ddsarsas = []
    targets = []
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except:
        pass

    for i in lrs:
        m1 = DeepDoubleSarsa(8, act_len, True, i)
        m1.to(device)
        m2 = DeepDoubleSarsa(8, act_len, True, i)
        m2.to(device)
        ddsarsas.append(m1)
        targets.append(m2)

    train_return = mp.Manager().dict()
    for e in range(1, EPOCH+1):
        processes = []
        start = time.time()
        for rank in range(num_processes):
            p = mp.Process(target=pbtSarsa, args=(rank, ddsarsas[rank], targets[rank], device, eps, e, train_return, False))
            p.start()
            processes.append(p)

        print("\n===========================================\n")
        print("Epoch {} - Training/Evaluation Sequences\n".format(e))
        for p in processes: p.join()

        print("\nEpoch {} - Exploration/Exploitation Sequences\n".format(e))
        bestPerformers, worstPerformers = exploitPopulation(train_return, best=BEST, worst=WORST)
        processes = []

        for rank in range(num_processes):
            p2 = mp.Process(target=exploreIndividual, args=(rank, ddsarsas[rank], targets[rank], bestPerformers, worstPerformers, train_return))
            p2.start()
            processes.append(p2)
        for p2 in processes: p2.join()
        stop = time.time()
        elapsedTime = stop - start
        leftTime = elapsedTime * (EPOCH - e - 1)
        print("Time spent on this epoch: {} mins {} secs".format(elapsedTime//60, elapsedTime%60))
        print("Expected time left for completion: {} mins {} sec".format(leftTime//60, leftTime%60))
        eps -= EPISODES*(epsilon_start - epsilon_stop)/epsilon_decay
    #* Save best model at the end
    for bestPerformer in bestPerformers:
        print("\nSaving best performer: Model {}.".format(bestPerformer))
        ddsarsas[bestPerformer].save("models/sarsa/target{}".format(bestPerformer))


if __name__ == "__main__":
    pbtRun()

