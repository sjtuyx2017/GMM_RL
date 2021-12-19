import numpy as np
import argparse
import os
import sys
import torch.nn.functional as F
from torch import nn,optim
import torch
from load_data import load_celebA_dataset,construct_data_loader,construct_data_loader2,get_data_loader
from models import EncoderModel,TopModel,DecoderModel,Actor,Critic
import matplotlib.pyplot as plt

#plt.switch_backend('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--device_id',  default='0', type=str)
parser.add_argument('--sigma',  default=1.0, type=float)
parser.add_argument('--cloud_task',  default='Smiling', type=str)
parser.add_argument('--attack_task',  default='Male', type=str)
parser.add_argument('--batch_size',  default=128, type=int)
parser.add_argument('--iteration_num',  default=20, type=int)
parser.add_argument('--local_epoch',  default=10, type=int)
parser.add_argument('--lr',  default=1e-3, type=float)
parser.add_argument('--target_acc',default=1.0,type=float)
parser.add_argument('--target_pri',default=0.5,type=float)

#reinforcement learning parameters
parser.add_argument('--episodes',  default=100, type=int)
parser.add_argument('--state_num',  default=5, type=int)
parser.add_argument('--capacity',default=10000,type=int)
parser.add_argument('--tau',  default=0.005, type=float)
parser.add_argument('--gamma',  default=0.99, type=float)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--update_iteration',  default=20, type=int)
parser.add_argument('--child_model_num',  default=5, type=int)

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create results saving path
directory = './GMM_RL_results'
if not os.path.exists(directory):
    os.mkdir(directory)
SavePath = directory + '/%s_%s' % (args.cloud_task, args.attack_task)
if not os.path.exists(SavePath):
    os.mkdir(SavePath)
modelSavePath = SavePath + '/models'
if not os.path.exists(modelSavePath):
    os.mkdir(modelSavePath)
figureSavePath = SavePath + '/figures'
if not os.path.exists(figureSavePath):
    os.mkdir(figureSavePath)

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r = [], [], [], []

        for i in ind:
            X, Y, U, R = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r)

class LocalModel(nn.Module):
    def __init__(self, encoder, top_model,actor,actor_target,critic,critic_target,train_loader,validation_loader):
        super(LocalModel, self).__init__()
        self.encoder = encoder.to(device)
        self.top_model = top_model.to(device)
        self.actor = actor.to(device)
        self.actor_target = actor_target.to(device)
        self.critic = critic.to(device)
        self.critic_target = critic_target.to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.sigma = args.sigma
        self.privacyLoss = privacyLoss(sigma=self.sigma)
        self.lr = args.lr
        self.optimizer_top = optim.Adam(self.top_model.parameters(), lr=self.lr)
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        # record the current state and the past few states
        self.recent_states = [0]* args.state_num *2
        self.all_states = []
        self.target_acc = args.target_acc
        self.target_pri = args.target_pri
        self.replay_buffer = Replay_buffer()
        self.cov = torch.eye(256)

    def get_gaussian(self, mu):
        sigma = 0.01
        if sigma == 0:
            return mu
        fx,fy = mu.shape
        # feature1 = feature.cpu().detach().numpy()
        temp = np.random.multivariate_normal([0 for i in range(fy)], self.cov)
        # Data are generated according to covariance matrix and mean
        for i in range(1, fx):
            temp = np.concatenate((temp, np.random.multivariate_normal([0 for i in range(fy)],self.cov)),axis=0)
            # Splicing sampling of high dimensional Gaussian distribution data
        temp.resize((fx, fy))
        temp = torch.from_numpy(temp).float()
        # Since the stitched data is one-dimensional,
        # we redefine it as the original dimension
        feature = mu + temp.to(self.device)* (sigma**0.5)
        return feature

    def select_action(self,states):
        states = torch.FloatTensor(states).to(device)
        action = self.actor(states).cpu().data.numpy()
        # add exploration noise to help the exploration
        action = (action + np.random.normal(0, args.exploration_noise, size=1)).clip(0,1)
        return action

    def get_state(self):
        batch_size = args.batch_size

        # prepare datasets
        decoder_train_loader,_ = get_data_loader(self.encoder,self.train_loader,batch_size,batch_size,0.01,device)
        decoder_val_loader,top_val_loader = get_data_loader(self.encoder,self.validation_loader,batch_size,batch_size,0.01,device)

        # test accuracy on validation set
        top_correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(top_val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.top_model(inputs)
                _, predicted = outputs.max(1)
                top_correct += predicted.eq(targets).sum().item()
        val_acc = top_correct/float(len(top_val_loader.dataset))

        # train decoder and test privacy on validation set
        decoder = DecoderModel(decoder_train_loader,decoder_val_loader)
        val_pri = decoder.train_decoder(50)

        # treat the target accuracy and privacy as (1,1) and calculate the relative value
        relative_acc = val_acc/self.target_acc
        relative_pri = val_pri/self.target_pri
        state = [relative_acc,relative_pri]
        self.all_states.append(state)
        del(self.recent_states[0])
        del(self.recent_states[0])
        self.recent_states.append(relative_acc)
        self.recent_states.append(relative_pri)
        return state

    def get_reward(self,current_state,next_state):
        current_d1 = (current_state[0]**2+current_state[1]**2)
        current_d2 = 0.5*(current_state[0]-current_state[1])**2
        next_d1 = (next_state[0]**2+next_state[1]**2)
        next_d2 = 0.5*(next_state[0]-next_state[1])**2

        if (next_d1 - next_d2) > (current_d1 - current_d2) and (next_d1/next_d2) > (current_d1/current_d2):
            reward = 1
        else:
            reward = -1
        return reward

    def update(self):
        for iteration in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


    def train_network(self):
        train_loader = self.train_loader
        total_episodes = args.episodes
        iteration_num = args.iteration_num

        privacy_loss_list = []
        accuracy_loss_list = []

        iter = train_loader.__iter__()
        batch_num = len(train_loader)

        next_state = self.get_state()

        for episode in range(total_episodes):
            current_state = next_state
            current_states = self.recent_states
            action = self.select_action(self.recent_states)
            weight = action
            # update the encoder for k iterations
            for iteration in range(iteration_num):
                try:
                    batch_data = iter.__next__()
                # end of the current data loader
                except StopIteration:
                    iter = train_loader.__iter__()
                    batch_data = iter.__next__()
                inputs, targets, private_targets = batch_data
                inputs, targets, private_targets = inputs.to(device), targets.to(device), private_targets.to(device)
                mu = self.encoder(inputs)
                features = self.get_gaussian(mu)
                outputs = self.top_model(features)

                acc_loss = self.criterion(outputs, targets.long())
                pri_loss = self.privacyLoss(mu, private_targets)

                privacy_loss_list.append(pri_loss.item())
                accuracy_loss_list.append(acc_loss.item())

                loss = weight * acc_loss + (1-weight) * pri_loss
                self.optimizer_top.zero_grad()
                self.optimizer_encoder.zero_grad()
                loss.backward()
                self.optimizer_top.step()
                self.optimizer_encoder.step()

            next_state = self.get_state()
            next_states = self.recent_states
            reward = self.get_reward(current_state,next_state)
            self.replay_buffer.push((current_states,next_states,action,reward))
            # update the actor and critic network
            self.update()

        torch.save(self.actor.state_dict(), modelSavePath + 'actor.pth')

        file = open('states.txt','w')
        file.write(str(self.all_states))
        file.close()

class privacyLoss(nn.Module):
    def __init__(self,sigma):
        super(privacyLoss,self).__init__()
        self.sigma = sigma

    def forward(self, feature, private_attribute):
        features = torch.split(feature, 1, dim=0)
        batch_size = private_attribute.shape[0]
        k = features[0].shape[1]

        keys = []
        mu_Fs = {}
        count_Fs = {}
        Sigma_Fs = {}
        Sigma_a = torch.eye(k).float().to(device) * self.sigma

        for i in range(batch_size):
            label = float(private_attribute[i])
            if label in mu_Fs:
                mu_Fs[label] += features[i].t()
                count_Fs[label] += 1
            else:
                mu_Fs[label] = features[i].t()
                count_Fs[label] = 1

        for i in range(batch_size):
            label = float(private_attribute[i])
            mu_f = mu_Fs[label] / count_Fs[label]
            if label in Sigma_Fs:
                Sigma_Fs[label] += torch.mm((features[i].t() - mu_f), (features[i].t() - mu_f).t())
            else:
                Sigma_Fs[label] = torch.mm((features[i].t() - mu_f), (features[i].t() - mu_f).t())

        result = torch.Tensor([0.0]).float().to(device)

        for key in mu_Fs:
            mu_Fs[key] = mu_Fs[key] / count_Fs[key]
            Sigma_Fs[key] = Sigma_a + Sigma_Fs[key] / count_Fs[key]
            keys.append(key)

        # 0.02
        # print(len(keys))
        # kltime = datetime.datetime.now()
        for i in range(len(keys) - 1):
            s1, u1 = Sigma_Fs[keys[i]], mu_Fs[keys[i]]
            for j in range(1, len(keys)):
                if j!=i:
                    s2, u2 = Sigma_Fs[keys[j]], mu_Fs[keys[j]]
                    result += (self.kldiv(s1, u1, s2, u2, k)[0] * count_Fs[keys[i]] * count_Fs[keys[j]])

        # kltime = datetime.datetime.now() - kltime
        # print(kltime)
        # 0.4-0.8

        return result/(batch_size**2)

    def kldiv(self, s1, u1, s2, u2, k):
        temp1 = (u1 - u2)
        s1_det = s1.det()
        s2_det = s2.det()
        tr = torch.trace(torch.mm(s2.inverse(), s1))
        temp2 = torch.mm(torch.mm(temp1.t(), s2.inverse()), temp1)[0][0]
        
        print("s1: ",s1)
        print("s2: ",s2)
        print("s1 det: ",s1_det)
        print("s2 det: ",s2_det)
        print("trace: ",tr)
        print("temp: ",temp2)
        
        result = 0.5 * (torch.log2(s2.det() / (s1.det()+1e-10 )+1e-10) - k
                        + torch.mm(torch.mm(temp1.t(), s2.inverse()), temp1) + 
                        torch.trace(torch.mm(s2.inverse(), s1)))
        
        print(result[0])
        if result[0] < 0:
            print("KL divergence < 0 !!!")
            print("KL divergence < 0 !!!")
            print("KL divergence < 0 !!!")
        print("\n")
        return result

def main():
    cloud_task = args.cloud_task
    attack_task = args.attack_task
    batch_size = args.batch_size
    print("cloud task: ", cloud_task)
    print("attack target: ", attack_task)

    state_dim = args.state_num * 2

    # prepare the data loader
    train_data, train_label, test_data, test_label, val_data, val_label, train_private_att, test_private_att, val_private_att = load_celebA_dataset(
        cloud_task, attack_task)
    train_loader = construct_data_loader2(train_data, train_label, train_private_att, batch_size)
    test_loader = construct_data_loader2(test_data, test_label, test_private_att, batch_size)
    validation_loader = construct_data_loader2(val_data, val_label, val_private_att, batch_size)

    # train policy network(actor network)
    encoder = EncoderModel()
    simulation_topmodel = TopModel()
    actor = Actor(state_dim)
    actor_target = Actor(state_dim)
    critic = Critic(state_dim)
    critic_target = Critic(state_dim)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    local_model = LocalModel(encoder, simulation_topmodel, actor, actor_target, critic, critic_target, train_loader,
                             validation_loader)
    local_model.train_network()
    print("policy network training over")

    print("cloud task: ", cloud_task)
    print("attack target: ", attack_task)

if __name__ == '__main__':
    main()
