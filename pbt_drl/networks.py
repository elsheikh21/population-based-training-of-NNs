import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, next_action, done, q1n, q2n):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, next_action, done, q1n, q2n)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, next_action, done, q1n, q2n = map(np.stack, zip(*batch))
        return state, action, reward, next_state, next_action, done, q1n, q2n
    
    def __len__(self):
        return len(self.buffer)


class DeepDoubleSarsa(torch.nn.Module):

    def __init__(self, initus, exitus, bias=False, lr=0.00025):
        super(DeepDoubleSarsa, self).__init__()
        
        self.fc1 = torch.nn.Linear(initus, 64, bias=bias)
        self.fc2 = torch.nn.Linear(64, 64, bias=bias)
        self.fc3 = torch.nn.Linear(64, exitus, bias=bias)
        
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, input):
        q = input
        q = torch.nn.functional.relu(self.fc1(q))
        q = torch.nn.functional.relu(self.fc2(q))
        q = self.fc3(q)

        return q

    def update(self, sarsa, q2, gamma):   
        s, a, r, sn, an, d = sarsa
        s = Variable(torch.FloatTensor(s)).cuda()
        r = Variable(torch.FloatTensor(r))
        d = Variable(torch.FloatTensor(d))
        qb = Variable(torch.FloatTensor(q2))

        q = self(s)

        in_q = [np.arange(len(a)), a]
        in_qb = [np.arange(len(an)), an]

        self.optimizer.zero_grad()
        loss = torch.mean(torch.pow(r + (1.0 - d)*gamma*qb[in_qb] - q.cpu()[in_q],2)/2.0)
        loss.backward()
        self.optimizer.step()
        return loss

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def save(self, filepath):
        torch.save({
            'model_state_dict' : self.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict()
        } , filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def perturb(self, worker, seed):
        np.random.seed(seed)
        for _,v in self.state_dict().items():
            v += Variable(torch.FloatTensor(np.random.normal(0.0, 0.001, size = v.shape))).cuda()

        #* Hyperparameter perturb
        lrChange = max(0.001, np.random.normal(0.0, 0.01))
        for g in self.optimizer.state_dict()['param_groups']:
            g['lr'] += lrChange
