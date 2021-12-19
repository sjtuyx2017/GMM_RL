import numpy as np
import os
import argparse
import sys
sys.path.append('/home/lxiang_stu5/yuxi/new_celebA/')
from torch import nn,optim
import torch
from load_data import load_celebA_dataset,construct_data_loader,construct_data_loader2,get_data_loader
from models import DecoderModel,CloudTask
import collections


parser = argparse.ArgumentParser()
parser.add_argument('--device_id',  default='0', type=str)
parser.add_argument('--cloud_task',  default='Smiling', type=str)
parser.add_argument('--attack_task',  default='Male', type=str)
parser.add_argument('--top_epoch',  default=20, type=int)
parser.add_argument('--decoder_epoch',  default=100, type=str)
parser.add_argument('--lam',  default=0.1, type=float)
parser.add_argument('--sigma',  default=0.01, type=str)
parser.add_argument('--top_model_batchsize',  default=128, type=int)
parser.add_argument('--decoder_batchsize',  default=128, type=int)

args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    cloud_task = args.cloud_task
    attack_task = args.attack_task
    top_model_batchsize = args.top_model_batchsize
    decoder_batchsize = args.decoder_batchsize
    top_epoch = args.top_epoch
    decoder_epoch = args.decoder_epoch
    sigma = args.sigma

    train_data, train_label, test_data, test_label, train_private_att, test_private_att = load_celebA_dataset(cloud_task,attack_task)
    criterion = torch.nn.CrossEntropyLoss()
    
    # load decoder
    #epoch = sys.argv[2]
    file_name = 'encoder_9.pkl'
    encoder = torch.load('../GMM/GMM_results/%s_%s_%s/models/'%(cloud_task,attack_task,str(lam))+ file_name)

    print(collections.Counter(np.array(test_private_att)))
    # get feature after the encoder is fixed
    #train_feature = encoder(train_data.to(device)).detach()
    #test_feature = encoder(test_data.to(device)).detach()

    # prepare the datasets
    train_loader = construct_data_loader2(train_data,train_label,train_private_att,128)
    test_loader = construct_data_loader2(test_data,test_label,test_private_att,128)

    print("processing training dataset")
    top_train_loader , decoder_train_loader = get_data_loader(encoder , train_loader,top_model_batchsize,decoder_batchsize,sigma,device)

    print("processing test dataset")
    top_test_loader , decoder_test_loader = get_data_loader(encoder , test_loader,top_model_batchsize,decoder_batchsize,sigma,device)

    top_model = CloudTask(top_train_loader,top_test_loader)
    decoder = DecoderModel(decoder_train_loader, decoder_test_loader)
    
    top_model.train_model(top_epoch)
    top_model.test_model()
    
    decoder.train_decoder(decoder_epoch)
    
    print("cloud task: ",cloud_task)
    print("attack task: ",attack_task)
    print("sigma: ",sigma)
    print("lambda: ",lam)
