import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

def load_celebA_dataset(label_name,private_attribute_name):

    print('cloud task: ',label_name)
    print('attack task: ',private_attribute_name)

    attribute_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    label_number = attribute_list.index(label_name)
    private_attribute_number = attribute_list.index(private_attribute_name)
    
    print("private attribute number: ",private_attribute_number)


    dataPath = '../data/'
    f1 = open(dataPath + 'train_data.pickle', 'rb')
    f2 = open(dataPath + 'train_label.pickle', 'rb')
    f3 = open(dataPath + 'test_data.pickle', 'rb')
    f4 = open(dataPath + 'test_label.pickle', 'rb')

    X_train = np.array(pickle.load(f1), dtype='float32')
    y_train = pickle.load(f2)
    X_test = np.array(pickle.load(f3), dtype='float32')
    y_test = pickle.load(f4)
    f1.close()
    f2.close()
    f3.close()
    f4.close()

    train_data = torch.from_numpy(X_train)
    train_label = torch.from_numpy(y_train[:,label_number])
    test_data = torch.from_numpy(X_test)
    test_label = torch.from_numpy(y_test[:,label_number])
    train_private_label = torch.from_numpy(y_train[:,private_attribute_number])
    test_private_label = torch.from_numpy(y_test[:,private_attribute_number])
    print("train label ratio: ",Counter(np.array(train_label)))
    print("test label ratio: ",Counter(np.array(test_label)))
    print("train private label ratio: ",Counter(np.array(train_private_label)))
    print("test private label ratio: ",Counter(np.array(test_private_label)))

    return train_data,train_label,test_data,test_label,train_private_label,test_private_label

# this data loader only contains data and accuracy label
def construct_data_loader(data,label,batch_size):
    dataset = TensorDataset(data, label)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader

# this data loader contains data , accuracy label and privacy label
def construct_data_loader2(data,label,private_label,batch_size):
    dataset = TensorDataset(data, label,private_label)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def get_feature(inputs,encoder,sigma,device):
    cov = torch.eye(256)
    mu = encoder(inputs.to(device))
    fx, fy = mu.shape
    # feature1 = feature.cpu().detach().numpy()
    temp = np.random.multivariate_normal([0 for i in range(fy)], cov)
    # Data are generated according to covariance matrix and mean
    for i in range(1, fx):
        temp = np.concatenate((temp, np.random.multivariate_normal([0 for i in range(fy)], cov)), axis=0)
        # Splicing sampling of high dimensional Gaussian distribution data
    temp.resize((fx, fy))
    temp = torch.from_numpy(temp).float()
    # Since the stitched data is one-dimensional,
    # we redefine it as the original dimension
    feature = mu + temp.to(device) * (sigma ** 0.5)
    return feature

# get the intermediate data loader for top model and decoder
def get_data_loader(encoder,loader,top_batchsize,decoder_batchsize,sigma,device):
    features = torch.tensor([])
    labels = torch.tensor([])
    private_attributes = torch.tensor([])
    for batch_idx, (inputs, targets, private_targets) in enumerate(loader):
        # send original data to the fixed encoder and get the intermediate features
        feature = get_feature(inputs, encoder,sigma,device).detach().to('cpu')
        features = torch.cat([features, feature], dim=0)
        labels = torch.cat([labels, targets], dim=0)
        private_attributes = torch.cat([private_attributes, private_targets], dim=0)

    decoder_loader = construct_data_loader(features, private_attributes, decoder_batchsize)
    top_model_loader = construct_data_loader(features, labels, top_batchsize)

    return decoder_loader, top_model_loader



