import argparse
import os
import os.path as osp
from copy import deepcopy
from re import A
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
from torch_geometric.nn import GCNConv

from torch.optim.lr_scheduler import MultiStepLR

from models.utils import ensure_path, progress_bar
from models.utils import pprint, ensure_path
from torch.distributions import Categorical 
import copy
import random
import json
import re
from rapidfuzz import fuzz

import logging
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split 
from models import GCN_model2, MLP, labelConv
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from data.gen_data import SingleGraphOFADataset
from utils import SentenceEncoder, filter_category_by_TFIDT, replace_category_by_annotation
from llm import generate_close_topics, efficient_openai_text_api, llm_1v1, \
    llm_llama_ood, generate_ood_prompts, get_result_from_ood, efficient_openai_text_ood

from openail.utils import load_mapping, load_mapping_2
import csv

device = ''
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def cal_center_loss(matrix, indices):
    center_all = torch.tensor([])
    center_all = center_all.to(device)
    center_loss = torch.tensor([0.])
    center_loss = center_loss.to(device)
    for i in args.known_class:
        new_indices = torch.where(g.original_y[indices] == i)[0]
        new_matrix = matrix[indices][new_indices]
        center = torch.mean(new_matrix, dim = 0)
        center = torch.unsqueeze(center, dim = 0)
        center_distance = new_matrix - center
        center_loss += torch.norm(center_distance)
        center_all = torch.cat([center_all, center], dim = 0)
    return center_all, center_loss / len(indices)
    
def find_margin_nodes_1(adj_matrix, indices):
    degree = torch.sum(adj_matrix, dim = 0)
    indices1 = torch.where(degree == 1)[0]
    new_indices = indices[indices1]
    return new_indices

def find_margin_nodes_2(net, seq, edge_index, indices, mixup_num = 100):
    res = net(seq, edge_index)[indices]
    values, indexs = torch.max(res, dim = 1) 
    indexs1, indexs2 = torch.sort(values)
    return indices[indexs2.cpu()[:mixup_num]]

def show_distribution(indices):
    labels = g.y[indices]
    label = torch.unique(labels)
    lens = []
    for item in label:
        length = len(torch.where(labels == item)[0])
        lens.append(length)
    return label, lens, len(labels)

def cal_cosine_distance(vector1, vector2):
    a = torch.sum(vector1 * vector2)
    vector1 = vector1 * vector1
    vector2 = vector2 * vector2
    b = torch.sqrt(torch.sum(vector1) * torch.sum(vector2))
    return 1. * a / b

def cal_E_distance(vector1, vector2):
    vector3 = vector1 - vector2
    vector4 = vector3 * vector3
    return torch.sqrt(torch.sum(vector4))

def cal_loss_5(logits, indices, margin = 0):
    res = torch.softmax(logits, dim = 1)
    loss = 0.
    for i in range(len(indices)):
        loss += res[i, new_labels[indices[i]]] - res[i, len(args.known_class)]
    loss /= len(indices)
    if loss < margin:
        loss = 0
    return loss

def get_neighbors(edge_index, target_nodes, n):
    # edge_index: shape [2, num_edges]
    # target_nodes: list or tensor of node indices
    neighbors = []

    # Convert to set for fast lookup
    target_nodes_set = set(target_nodes)

    for src, dst in edge_index.t():
        if src.item() in target_nodes_set:
            neighbors.append(dst.item())
        elif dst.item() in target_nodes_set:
            neighbors.append(src.item())
    
    # move duplicates
    neighbors = list(set(neighbors) - target_nodes_set)

    selected_neighbors = random.sample(neighbors, min(n, len(neighbors)))
    return selected_neighbors

def traindummy(epoch, net, optimizer, criterion, g, train_indices, selected_unseen_indices, mixup_num):

    print('\n Epoch: %d' % epoch)
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = args.alpha

    optimizer.zero_grad()
    beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        
    # pre_indices, later_indices = train_test_split(train_indices, test_size = 0.5)
    # random.shuffle(train_indices)
    # pre_indices, later_indices = train_indices[:len(train_indices) // 2], train_indices[len(train_indices) // 2:]
    seq = g.x
    edge_index = g.edge_index
    num_old_class = seq.shape[0]
    # pre2embeddings = pre2block(net, seq, edge_index)
    # seq_new = torch.cat([seq, new_class_embeddings], dim = 0)
    new_edge_index = deepcopy(edge_index)
    delete_edge_index_all = torch.tensor([]).to(device)
    for i in range(unseen_indices.shape[0]):
        delete_edge_index1 = torch.where(new_edge_index[0]==unseen_indices[i])[0]
        delete_edge_index2 = torch.where(new_edge_index[1]==unseen_indices[i])[0]
        delete_edge_index_all = torch.cat([delete_edge_index_all, delete_edge_index1.long()], dim = 0)
        delete_edge_index_all = torch.cat([delete_edge_index_all, delete_edge_index2.long()], dim = 0)
    
    mask = torch.ones(new_edge_index.size(1), dtype=torch.bool)  # Create a mask for all edges
    mask[delete_edge_index_all.long()] = False  # Set the edges to delete as False
    new_edge_index = new_edge_index[:, mask]  # Select only the edges that are True in the mask
    
    if args.fine_method in ['denoising', 'denoising_mixup', 'graph_mixup']:
        one_hot_labels = torch.zeros(g.y.shape[0], len(args.known_class)+1).cuda()
        selected_indices = np.concatenate([train_indices, selected_unseen_indices])
        y_selected_indices = torch.cat([g.y[train_indices], torch.ones(len(selected_unseen_indices)).long().cuda() * len(args.known_class)], dim = 0)
        one_hot_labels[selected_indices] = F.one_hot(y_selected_indices, num_classes=len(args.known_class)+1).float()
        
        De = labelConv()
        lp_labels = De.label_propagate(one_hot_labels, new_edge_index, train_indices, selected_unseen_indices, order=3, drop_rate=0)
        # remove_indices = torch.where(lp_labels[selected_unseen_indices].max(dim=1)!=len(args.known_class)+1)[0]
        fine_labels = lp_labels[selected_unseen_indices].max(dim=1)[1]
        reserve_indices = torch.where(fine_labels==len(args.known_class))[0]
        selected_unseen_indices = selected_unseen_indices[reserve_indices.cpu().numpy()]
        
    pre2embeddings = pre2block(net, seq, new_edge_index)
    new_mixed_embeddings = torch.tensor([]).to(device)
    if args.fine_method in ['mixup', 'denoising_mixup']:
        new_class_embeddings = pre2embeddings[selected_unseen_indices]
        center_new_class = torch.mean(new_class_embeddings, dim = 0)
        num_new_class = new_class_embeddings.shape[0]
        
        new_parents = torch.tensor([]).to(device)
        # node_num = g.x.shape[0]
        # centers_all, center_loss = cal_center_loss(pre2embeddings, train_indices)
        
        # margin_nodes_indices1 = find_margin_nodes_1(train_adj, pre_indices)
        margin_nodes_indices = find_margin_nodes_2(net, seq, edge_index, train_indices, mixup_num)    


        cos_distance_p_max = [] # store the cosine distance between p1 and mixed point

        
        # mixed_embedding = torch.cat([mixed_embedding, g.x[unseen_indices[:30]]], dim = 0)
        for i in range(len(margin_nodes_indices)):
            for j in range(1):
                p1 = margin_nodes_indices[i]
                mixed_embedding = beta * pre2embeddings[p1] + (1-beta) * center_new_class
                mixed_embedding = mixed_embedding.unsqueeze(0)
                new_mixed_embeddings = torch.cat([new_mixed_embeddings, mixed_embedding], dim = 0)
                # new_parents = torch.cat([new_parents, torch.tensor([[-1], [p1]]).to(device)], dim = 1)
                distance = cal_cosine_distance(pre2embeddings[p1], mixed_embedding)
                cos_distance_p_max.append(distance)

                    
        pre2embeddings = torch.cat([pre2embeddings[:num_old_class], new_mixed_embeddings], dim = 0)
    elif args.fine_method in['graph', 'graph_mixup']:
        new_class_embeddings = pre2embeddings[selected_unseen_indices]
        center_new_class = torch.mean(new_class_embeddings, dim = 0)
        neighbors_indices = get_neighbors(new_edge_index, selected_unseen_indices, mixup_num)
        for i in range(len(neighbors_indices)):
            p1 = neighbors_indices[i]           
            mixed_embedding = beta * pre2embeddings[p1] + (1-beta) * center_new_class
            mixed_embedding = mixed_embedding.unsqueeze(0)
            new_mixed_embeddings = torch.cat([new_mixed_embeddings, mixed_embedding], dim = 0)
            # new_parents = torch.cat([new_parents, torch.tensor([[i], [-1]]).to(device)], dim = 1)
        pre2embeddings = torch.cat([pre2embeddings[:num_old_class], new_mixed_embeddings], dim = 0)
            
        
    # train_indices = np.concatenate([train_indices, selected_unseen_indices])
    sum = new_mixed_embeddings.shape[0]
    print(new_mixed_embeddings.shape)
    
        
    # sum = new_sum
    # sum = len(selected_unseen_indices)
    print(f'sum of new point :{sum}')


    
    new_output = latter2blockclf1(net, pre2embeddings, new_edge_index)

    loss7 = torch.tensor([0]).cuda()

    true_labels = torch.cat([new_labels[train_indices], (torch.ones(selected_unseen_indices.shape[0]) * len(args.known_class)).long().cuda(), (torch.ones(sum) * len(args.known_class)).long().cuda()])
    train_indices = np.concatenate([train_indices, selected_unseen_indices])
    if sum == 0:
        predicted_outputs = new_output[train_indices]
    else:
        predicted_outputs = torch.cat([new_output[train_indices], new_output[-sum:]], dim = 0)
    loss = criterion(predicted_outputs, true_labels)

    loss.backward()
    optimizer.step()
                   
    _, predicted = predicted_outputs.max(1)
    total = len(train_indices) + sum
    new_labels_gpu = new_labels.to(device)
    print(type(new_labels_gpu))
    correct = predicted.eq(true_labels).sum().item()
    
    if sum == 0:
        train_open_output = torch.max(new_output[selected_unseen_indices], dim = 1)[1]
    else:
        train_open_output = torch.max(torch.cat([new_output[selected_unseen_indices], new_output[-sum:]],dim=0), dim = 1)[1]
    train_open_correct = int(train_open_output.eq((torch.ones(selected_unseen_indices.shape[0]+sum) * len(args.known_class)).long().cuda()).sum()) / (selected_unseen_indices.shape[0]+sum)
    print(f'train_open_correct = {train_open_correct}')

    if args.shmode == False:
        print(f'epoch = {epoch}, correct = {correct}, total = {total}')
        print(f'Acc : {correct*1. / total}')
        print(f'L1 : {loss.item()}')
        # print(f'L2 : {loss2.item()}')
        # print(f'L3 : {loss3.item()}')
        # print(f'L4 : {loss4.item()}')
        print(f'L7 : {loss7.item()}')

def complement_loss(preds, targets):
    """
    preds: Tensor of shape (batch_size, C+1) containing the predicted probabilities for each class.
    targets: Tensor of shape (batch_size,) containing the true class indices for each sample.
    """

    batch_size, num_classes = preds.size()
    
    # Prepare a mask to ignore the correct class in summation
    mask = torch.eye(num_classes, dtype=torch.bool, device=preds.device)[targets]
    
    preds = torch.softmax(preds, dim=1)
    # Predicted probability for the correct class
    y_hat_yi = preds.gather(1, targets.view(-1, 1)).squeeze()

    loss = 0
    for i in range(batch_size):
        loss_item = 0
        for c in range(num_classes):
            if c != targets[i]:  # skip the true class
                # Calculate the loss component for class c
                y_hat_ic = preds[i, c]
                loss_item += (y_hat_ic / (1 - y_hat_yi[i])) * torch.log(y_hat_ic / (1 - y_hat_yi[i]))
        loss += -loss_item
    
    # Final loss is the negative summation
    loss = loss / batch_size
    return loss
        
def cal_max(array):
    data = torch.tensor(array)
    data = torch.unsqueeze(data, dim = 0)
    max, epoch = torch.max(data, dim = 1)
    return max, epoch

def train(epoch, net, optimizer, criterion, g, train_indices, selected_unseen_indices):
        
    print('\n Epoch: %d' % epoch)
    # net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    seq = g.x
    edge_index = g.edge_index
    optimizer.zero_grad()
    outputs = net(seq, edge_index)
    train_indices = np.concatenate([train_indices, selected_unseen_indices])
       
    # loss1 = criterion(outputs[selected_unseen_indices], (torch.ones(len(selected_unseen_indices)) * len(args.known_class)).long().cuda())
    # loss2 = criterion(outputs[train_indices], g.y[train_indices].long().cuda())
    # loss = loss1 + loss2
    loss = criterion(outputs[train_indices], g.y[train_indices].long().cuda())
    loss.backward()
    optimizer.step()
    _, predicted = outputs[train_indices].max(1)
    total = len(train_indices)
    new_labels_gpu = g.y.to(device)
    print(type(new_labels_gpu))
    correct = predicted.eq(new_labels_gpu[train_indices]).sum().item()
    
    train_open_output = torch.max(outputs[selected_unseen_indices], dim = 1)[1]
    num_new_point = len(selected_unseen_indices)
    train_open_correct = int(train_open_output.eq((torch.ones(num_new_point) * len(args.known_class)).long().cuda()).sum()) / num_new_point

    if args.shmode == False:
        print(f'epoch = {epoch}, train_open_correct = {train_open_correct}, train_close_correct = {correct/total}, total = {total}')
        # print(f'Acc : {correct*1. / total}')
        print(f'loss : {loss.item()}')

        
def test(epoch, net, valid_closeset_indices, valid_openset_indices, test_closeset_indices, test_openset_indices, criterion):
    net.eval()
    seq = g.x
    edge_index = g.edge_index
    outputs = net(seq, edge_index)    

    close_outputs = outputs[valid_closeset_indices]
    open_outputs = outputs[valid_openset_indices]
    
    close_correct = torch.max(close_outputs, dim = 1)[1]
    valid_close_ratio = close_correct.eq(new_labels[valid_closeset_indices].to('cuda')).sum() / len(valid_closeset_indices)   
     
    open_outputs = torch.softmax(open_outputs, dim = 1)
    open_correct = torch.max(open_outputs, dim = 1)[1]
    open_true = torch.ones(len(valid_openset_indices)) * len(args.known_class)
    valid_open_ratio = open_correct.eq(open_true.to('cuda')).sum() / len(valid_openset_indices)
    
    valid_overall_ratio = (close_correct.eq(new_labels[valid_closeset_indices].to('cuda')).sum() + open_correct.eq(open_true.to('cuda')).sum()) / (len(valid_openset_indices) + len(valid_closeset_indices))
    
    valid_loss = criterion(outputs, new_labels.to('cuda'))
    
    close_outputs = outputs[test_closeset_indices]
    open_outputs = outputs[test_openset_indices]
    
    close_correct = torch.max(close_outputs, dim = 1)[1]
    test_close_ratio = close_correct.eq(new_labels[test_closeset_indices].to('cuda')).sum() / len(test_closeset_indices)
    
    open_outputs = torch.softmax(open_outputs, dim = 1)
    open_correct = torch.max(open_outputs, dim = 1)[1]
    open_true = torch.ones(len(test_openset_indices)) * len(args.known_class)
    test_open_ratio = open_correct.eq(open_true.to('cuda')).sum() / len(test_openset_indices)
    
    test_overall_ratio = (close_correct.eq(new_labels[test_closeset_indices].to('cuda')).sum() + open_correct.eq(open_true.to('cuda')).sum()) / (len(test_openset_indices) + len(test_closeset_indices) )
    
    test_loss = criterion(outputs, new_labels.to('cuda'))
    
    print(f'valid_close_ratio = {valid_close_ratio}, valid_open_ratio = {valid_open_ratio}, valid_ovrall_ratio = {valid_overall_ratio}, loss = {valid_loss.item()}')
    print(f'test_close_ratio = {test_close_ratio}, test_open_ratio = {test_open_ratio}, test_overall_ratio = {test_overall_ratio}, loss = {test_loss.item()}')
    valid_acc = valid_overall_ratio
    predicted_labels = torch.max(outputs, dim = 1)[1]
    # print(f'valid acc {acc}')
    # if acc > best_acc:
    #     print('Saving..')   
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     torch.save(state, osp.join(args.save_path,'ckpt.pth'))
    #     best_acc = acc
    return valid_acc, test_close_ratio, test_open_ratio, test_overall_ratio, predicted_labels

def getmodel(args):
    print('==> Building model..')
    if args.backbone == "GCN_model2":
        net = GCN_model2(args.dim_feat, 512, 128, len(args.known_class)+1)
    elif args.backbone == 'MLP':
        net = MLP(g.x.shape[1], 512, 128, len(args.known_class))
    net = net.cuda()
    return net

def finetune_proser(epoch = 59):
    print('Now processing epoch', epoch)

    net = getmodel(args)
    print('==> Resuming from checkpoints..')
    print(model_path)
    assert os.path.isdir(model_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = args.lr , momentum = 0.9, weight_decay = 5e-4)
    # modelname = 'Modelof_Epoch' + str(epoch) + '.pth'
    # print(osp.join(model_path, save_path2, modelname))
    # checkpoint = torch.load(osp.join(model_path, save_path2, modelname))

    # net.load_state_dict(checkpoint['net'])
    # if args.backbone == 'GCN_model2':
    #     net.clf2 = nn.Linear(128, args.dummynumber)
    # elif args.backbone == 'MLP':
    #     net.clf2 = nn.Linear(128, args.dummynumber)
    net = net.cuda()

    Finetune_MAX_EPOCH = args.epoch
    wholebestacc = 0
    
    num_negative=train_indices.shape[0]//len(args.known_class) * args.negative_ratio
    temperature = args.temperature
    key= args.key

    best_acc = 0  
    best_epoch = 0
    for finetune_epoch in range(Finetune_MAX_EPOCH):
        traindummy(finetune_epoch, net, optimizer, criterion, g, train_indices, selected_unseen_indices, args.mixup_num)

        val_acc, _, _, _, predicted_labels = test(finetune_epoch, net, valid_closeset_indices, valid_openset_indices, test_closeset_indices, test_openset_indices, criterion)
        if val_acc > best_acc:
            print('Saving model')
            best_acc = val_acc
            best_epoch = epoch
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(save_path, 'model_final.pth'))


    print("Testing model")
    print('Loading {}th epoch'.format(best_epoch))
    net.load_state_dict(torch.load(os.path.join(save_path, 'model_final.pth'))['state_dict'])
    _, test_close_acc, test_open_acc, test_overall_acc, predicted_labels = test(epoch, net, valid_closeset_indices, valid_openset_indices, test_closeset_indices, test_openset_indices, criterion)
    torch.save(predicted_labels, os.path.join(save_path, 'predicted_labels.pth'))
    
    return test_close_acc.item(), test_open_acc.item(), test_overall_acc.item(), predicted_labels
    
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)        

def dummypredict(net, seq, edge_index):
    if args.backbone == 'GCN_model2':
        out = net.Conv1(seq, edge_index)
        out = net.Conv2(out, edge_index)
        out = net.clf2(out) 
    elif args.backbone == 'MLP':
        out = net.Conv1(seq)
        out = net.Conv2(out)
        out = net.clf2(out)
    return out

def adjacency_matrix(edge_index, node_num):
    adjacency_matrix = torch.zeros(node_num, node_num)
    length = edge_index.shape[1]
    for i in range(length):
        x = edge_index[0][i]
        y = edge_index[1][i]
        adjacency_matrix[x,y] = 1
    return adjacency_matrix

def relabel_new(data, args, old_labels):
    old_label = torch.unique(old_labels)
    new_labels = torch.ones_like(old_labels)*100
    new_label = deepcopy(old_label)
    seen_label = torch.tensor(args.known_class)
    unseen_label = torch.tensor(np.setdiff1d(new_label.numpy(), seen_label.numpy()))

    seen_indices = []
    unseen_indices = []
    for label in unseen_label:
        indices = np.where(old_labels == label)[0]
        # new_labels[indices] = unseen_label[0]
        unseen_indices.append(indices)
    unseen_indices = np.concatenate(unseen_indices, axis=0)
    seen_indices = np.setdiff1d(np.arange(len(old_labels)), unseen_indices)
    # unseen_indices = np.where(new_labels == args.known_class)[0]

    for i in range(len(args.known_class)):
        new_labels[old_labels == seen_label[i]] = i
    new_labels[unseen_indices] = len(args.known_class)
    data.y = new_labels
    data.original_y = old_labels

    seen_train_indices, seen_test_indices = train_test_split(seen_indices, test_size = 1 - args.train_rate, random_state = 100)
    
    train_indices = seen_train_indices
    test_valid_indices = np.concatenate([seen_test_indices, unseen_indices], axis=0)
    valid_indices, test_indices = train_test_split(test_valid_indices, test_size = args.valid_rate / (1 - args.train_rate), random_state = 100)
    return data, seen_label, unseen_label, train_indices, valid_indices, test_indices, unseen_indices, seen_indices

def pre2block(net, seq, edge_index):
    if args.backbone == 'GCN_model2':
        out = net.Conv1(seq, edge_index)
    elif args.backbone == 'MLP':
        out = net.Conv1(seq)
    return out

def latter2blockclf1(net, seq, edge_index):
    if args.backbone == 'GCN_model2':
        out = net.Conv2(seq, edge_index)
        out = net.clf1(out)
    elif args.backbone == 'MLP':
        out = net.Conv2(seq)
        out = net.clf1(out)
    return out

def latter2blockclf2(net, seq, edge_index):
    if args.backbone == 'GCN_model2':
        out = net.Conv2(seq, edge_index)
        out = net.clf2(out)
    elif args.backbone == 'MLP':
        out = net.Conv2(seq)
        out = net.clf2(out)
    return out


def Conv1toConv2(net, seq, edge_index):
    if args.backbone == 'GCN_model2':
        out = net.Conv2(seq, edge_index)
    
    return out

def pretoConv2(net, seq, edge_index):
    out = net.Conv1(seq, edge_index)
    out = net.Conv2(out, edge_index)
    return out

def decompose(args, indices):
    indexs = []

    indices = torch.tensor(indices)

    for label in unseen_label:
        index = np.where(g.original_y[indices] == label)[0]
        index = list(index)
        indexs = indexs + index

    old_indexs = list(np.arange(len(indices)))
    new_indexs = list(set(old_indexs) - set(indexs))

    new_indexs = torch.tensor(new_indexs, dtype = torch.long)
    indexs = torch.tensor(indexs, dtype = torch.long)

    closeset_indices = indices[new_indexs]
    openset_indices = indices[indexs]

    return closeset_indices, openset_indices

if __name__=="__main__":
    # 构建参数列表
    parser = argparse.ArgumentParser(description = 'PyTorch Cora Training')
    parser.add_argument('--dataset', default = 'cora')
    parser.add_argument('--lr', default = 0.01, type = float, help = 'learning rate')
    parser.add_argument('--model_type', default = 'llm', type = str, help = 'Recognition Method')
    parser.add_argument('--known_class', default = [0,2,3,5], type = int, help = 'number of known class')
    parser.add_argument('--backbone', default = 'GCN_model2', type = str)
    parser.add_argument('--seed', default = '100', type = int, help = 'random seed for dataset generation')
    parser.add_argument('--lamda', default = '1', type = float, help = 'trade-off between loss')
    parser.add_argument('--lamda2', default = '1', type = float, help = 'trade-off between loss')
    parser.add_argument('--alpha', default = '1', type = float)
    parser.add_argument('--dummynumber', default = 1, type = int, help = 'number of dummy label.')
    parser.add_argument('--shmode', action = 'store_true')
    parser.add_argument('--ratio1', default = 0.5, type = float)
    parser.add_argument('--train_rate', default = 0.5)
    parser.add_argument('--valid_rate', default = 0.3)
    parser.add_argument('--epoch', default = 200)
    
    #data configs
    parser.add_argument('--key', default = 'change to your key', type = str)
    parser.add_argument('--llm_name', default = 'e5', type = str)
    parser.add_argument('--llm_b_size', default = 1, type = int)
    parser.add_argument('--negative_ratio', default = 1, type = int)
    parser.add_argument('--temperature', default = 0.1, type = float)
    parser.add_argument('--prompt_method', default = 'zero_shot_close', type = str)
    parser.add_argument('--llm_model_method', default = 'gpt-4o', type = str)
    parser.add_argument('--num_llm_dummy_class', default = 1, type = int)
    parser.add_argument('--seen_unseen_classifier', default = 'llm_train', type = str)
    parser.add_argument('--ood_classifier', default = 'none', type = str)
    parser.add_argument('--fine_method', default = 'denoising_mixup', type = str)
    parser.add_argument('--select_num', default = 50, type = int)
    parser.add_argument('--mixup_num', default = 100, type = int)
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    pprint(vars(args)) # print all the parameters
    # if args.dataset == 'cora':
    #     graph = Planetoid(root = '/data/projects/punim1970/iclr2025/G2Pxy/datasets', name = 'Cora')
    # g = graph[0]
    
    full_mapping = load_mapping_2()
    encoder = SentenceEncoder(args.llm_name, root=".", batch_size=args.llm_b_size)
    root = "cache_data"
    if args.llm_name != "ST":
        root = f"cache_data_{args.llm_name}"    
    if args.dataset in ['cora', 'citeseer', 'dblp', 'pubmed', 'wikics', 'elecomp', 'elephoto', 'arxiv']:
        # graph = Planetoid(root = '/data/projects/punim1970/iclr2025/G2Pxy/datasets', name = 'Cora')
        ofa_data = SingleGraphOFADataset(args.dataset, root=root, encoder=encoder)
    # g = graph[0]
    g = ofa_data.data
    g.x = g.node_text_feat
    g.original_y = g.y
    if args.dataset != 'arxiv':
        sorted_edges = torch.sort(g.edge_index, dim=0)[0]
        unique_edges = torch.unique(sorted_edges.T, dim=0)
        g.edge_index = unique_edges.T
        g.edge_index = torch.cat([g.edge_index, g.edge_index.flip(0)], dim=1)
        
    if args.dataset == 'arxiv':
        label = torch.tensor([0, 1, 2, 3, 4])  # Convert label list to tensor
        index = torch.where(torch.isin(g.y, label))[0]  # Get valid node indices

        # Create a mapping from old indices to new contiguous indices
        index_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(index)}

        # Filter node-related data
        g.y = g.y[index].view(-1)
        g.original_y = g.original_y[index].view(-1)
        g.x = g.x[index]
        g.raw_texts = [g.raw_texts[i] for i in index.tolist()]  # Ensure raw_texts remains a list

        # Filter edges where both source and target nodes are in `index`
        mask = torch.isin(g.edge_index[0], index) & torch.isin(g.edge_index[1], index)
        filtered_edges = g.edge_index[:, mask]

        # Remap edge indices to the new index range
        remapped_edges = torch.tensor([[index_map[i.item()] for i in filtered_edges[0]],
                                    [index_map[i.item()] for i in filtered_edges[1]]], dtype=torch.long)

        g.edge_index = remapped_edges

        # Ensure undirected edges
        g.edge_index = torch.cat([g.edge_index, g.edge_index.flip(0)], dim=1)

        # Load label names and filter based on `label`
        g.label_names = pd.read_csv('/data/projects/punim1970/iclr2025/TSGFM/data/single_graph/arxiv/labelidx2arxivcategeory.csv.gz').iloc[:, 1].tolist()
        g.label_names = [g.label_names[i] for i in label.tolist()]
        
    if args.dataset in ['elecomp', 'elephoto']:
        label_to_category = {}
        i = 0
        for label in g.y:
            label_str = str(label.item()) 
            if label_str not in label_to_category:
                label_to_category[label_str] = g.category_names[i]
            i += 1
            if len(label_to_category) == g.y.max()+1: 
                break
        g.label_names = [value for key, value in sorted(label_to_category.items(), key=lambda x: int(x[0]))]
    if args.dataset == 'dblp':
        g.label_names = ['Database', 'Data Mining', 'Artificial Intelligence', 'Information Retrieval']  
        # ['Information Retrieval', 'Database', 'AI', 'Data Mining']  
        g.category_names = [g.label_names[g.y[i]] for i in range(g.y.shape[0])]
    try:
        g.label_fullnames = [full_mapping[args.dataset][x] for x in g.label_names]
        g.label_fullnames = [x.lower() for x in g.label_fullnames]
        g.category_names = [g.label_fullnames[g.y[i]] for i in range(g.y.shape[0])]     
    except:
        print("No full mapping")
        g.label_fullnames = g.label_names
    
    ### The number of OOD classes is 2   
    if args.dataset == 'cora':
        args.known_class = [1,3,4,5,6]
    elif args.dataset == 'citeseer':
        args.known_class = [0,2,3,5]
    elif args.dataset == 'wikics':
        args.known_class = [0,1,2,3,4,5,6,7]
    elif args.dataset == 'dblp':
        args.known_class = [0,1]
    elif args.dataset == 'elecomp':
        args.known_class = [0,3,4,5,6,7,8,9]
    elif args.dataset == 'elephoto':
        args.known_class = [0,1,2,3,4,5,6,9,10,11]
    elif args.dataset == 'arxiv':
        args.known_class = [0,1,2]
    elif args.dataset == 'pubmed':
        args.known_class = [0]

    print("adj begin`")
    # adj = adjacency_matrix(g.edge_index, g.x.shape[0])
    print("adj end")
    node_num =g.y.shape[0]
    edge_index = g.edge_index
    # adjacent_matrix = adjacency_matrix(edge_index, node_num)
    # edge_num = torch.sum(adjacent_matrix)
    args.dim_feat = g.x.shape[1]
    g, seen_label, unseen_label, train_indices, valid_indices, test_indices, unseen_indices, seen_indices = relabel_new(g, args, g.original_y)
    # args.mixup_num = int(len(train_indices) / len(args.known_class))
    

    valid_closeset_indices, valid_openset_indices = decompose(args, valid_indices) #
    test_closeset_indices, test_openset_indices = decompose(args, test_indices) #

    g = g.to(device)
    new_labels = g.y
    save_path1 = osp.join('results', 'D{}-M{}-B{}-C'.format(args.dataset, args.model_type, args.backbone,)) 
    model_path = osp.join('results', 'D{}-M{}-B{}-C'.format(args.dataset, args.model_type, args.backbone,))
    save_path2 = 'K{}-U{}-E{}-Seed{}'.format(seen_label, unseen_label, str(args.llm_name), str(args.seed))
    args.save_path = osp.join(save_path1, save_path2)
    print(model_path)
    ensure_path(save_path1, remove = False)
    ensure_path(args.save_path, remove = False)

    # coarse-classifier    
    if args.seen_unseen_classifier == 'llm_train':
        if args.dataset in ['elecomp', 'elephoto']:
            topic = 'Amazon products'
        elif args.dataset in ['cora', 'citeseer', 'dblp', 'pubmed', 'wikics', 'arxiv']:
            topic = 'paper topic'
        if not osp.exists(args.dataset+"_"+str(args.known_class)+"_"+str(args.llm_model_method)+"_topics_unseen.json"):    
            output_data = generate_close_topics(g, topic, args, args.key)
            major_category_seen = output_data["major_category_seen"]
            dic_topics_unseen = output_data["dic_topics_unseen"]    

        else:
            output_data = json.load(open(args.dataset+"_"+str(args.known_class)+"_"+str(args.llm_model_method)+"_topics_unseen.json"))
            major_category_seen = output_data["major_category_seen"]
            dic_topics_unseen = output_data["dic_topics_unseen"]    
        topics_unseen = [answer["answer"] for answer in dic_topics_unseen]
        for i in range(len(topics_unseen)):
            match = re.search(r'^(.*):', topics_unseen[i])
            if match:
                topics_unseen[i] = match.group(1)
                
        if len(g.label_names)-len(args.known_class) == 2:  
            llm_answer_list, llm_confidence_list, llm_category_list, selected_test_indices = llm_1v1(g, topic, args, major_category_seen, topics_unseen, unseen_indices, seen_indices, test_indices)
        else:
            llm_answer_list, llm_confidence_list, llm_category_list, selected_test_indices = llm_1v1(g, args, major_category_seen, topics_unseen, unseen_indices, seen_indices, np.array(new_test))
        
        y_llm = [1 if val == 'True' or val == "true" else 0 for val in llm_answer_list]
        y_llm = np.array(y_llm)
        y_confidence = [float(val) for val in llm_confidence_list]
        y_confidence = np.array(y_confidence)
        
        y_test = []
        test_unseen_indices = []
        test_seen_indices = []
        for i in range(selected_test_indices.shape[0]):
            if selected_test_indices[i] in unseen_indices:
                y_test.append(0)
                test_unseen_indices.append(i)
            else:
                y_test.append(1)
                test_seen_indices.append(i)
        y_test = np.array(y_test)
        
        llm_accuray = (y_llm == y_test).sum() / len(y_test)
        test_unseen_accuray = (y_llm[test_unseen_indices] == 0).sum() / (len(y_test)-y_test.sum())
        categories = llm_category_list
        for i in range(len(llm_category_list)):
            categories[i] = re.sub(r'_', ' ', categories[i])
            categories[i] = re.sub(r'-', ' ', categories[i])
            categories[i] = categories[i].lower()
        for i in range(len(categories)):
            categories[i] = re.sub(r'case_based_reasoning', 'case_based', categories[i])
            categories[i] = re.sub(r'agents', 'Agent', categories[i])
        if args.dataset in ['cora', 'dblp', 'wikics']:   
            true_categories = [category.lower() for category in np.array(g.category_names)[test_indices]]
        else:
            true_categories = [category.lower() for category in np.array(g.category_names)[selected_test_indices]]
        test_seen_accuracy = (np.array(categories)[test_seen_indices] == np.array(true_categories)[test_seen_indices]).sum() / len(test_seen_indices)
        overall_acc = (test_unseen_accuray * len(test_unseen_indices) + test_seen_accuracy * len(test_seen_indices)) / len(test_indices)
        print( f'Test seen accuracy = {test_seen_accuracy}', f'Test unseen accuracy = {test_unseen_accuray}',  f'Overall accuracy = {overall_acc}')
        
        predicted_open_list = np.where(y_llm==0)[0]
        llm_open_accuracy = (y_llm[predicted_open_list]==y_test[predicted_open_list]).sum()/predicted_open_list.shape[0]
        
        predicted_close_list = np.where(y_llm==1)[0]
        llm_close_accuracy = (y_llm[predicted_close_list]==y_test[predicted_close_list]).sum()/predicted_close_list.shape[0]
        print(f'LLM accuracy = {llm_accuray}', f'Test unseen accuracy = {test_unseen_accuray}', f'LLM open accuracy = {llm_open_accuracy}')
    
        
        sorted_indices = np.argsort(y_confidence[predicted_open_list])[::-1]
        high_confidence_open_list1 = predicted_open_list[sorted_indices][:int(1*len(predicted_open_list))]
        selected_indices = np.where(y_confidence[predicted_open_list] > 0.5)[0]
        high_confidence_open_list2 = predicted_open_list[selected_indices]
        high_confidence_open_list = np.intersect1d(high_confidence_open_list1, high_confidence_open_list2)
        selected_unseen_indices = test_indices[high_confidence_open_list]
        # if not osp.exists(args.dataset+str(args.known_class)+'_selected_unseen_indices.npy'):
        np.save(args.dataset+'_selected_unseen_indices.npy', selected_unseen_indices)
        
        # selected_accuracy = (y_llm[high_confidence_open_list1] == y_test[high_confidence_open_list1]).sum() / len(high_confidence_open_list1)
        # print(f'Selected accuracy = {selected_accuracy}')
        
        unseen_category = np.array(llm_category_list)[high_confidence_open_list1]
        for i in range(len(unseen_category)):
            unseen_category[i] = re.sub(r'-', ' ', unseen_category[i])
            unseen_category[i] = re.sub(r'_', ' ', unseen_category[i])
            unseen_category[i] = unseen_category[i].capitalize()
        
        category_count = Counter(unseen_category)
        categories_all = [category for category, count in category_count.items()]
        counts_all = [category_count[i] for i in categories_all]
        # filtered_categories, filtered_counts = filter_category_by_TFIDT(categories_all, counts_all)
        
        most_common_category = max(category_count, key=category_count.get)
        most_common_category_count = category_count[most_common_category]
        # for category, count in category_count.items():
        categories_above_10 = [category for category, count in category_count.items() if count > 10]   
        counts_above_10 = [category_count[i] for i in categories_above_10]
        
        filtered_categories_above_10, filtered_counts_above_10 = filter_category_by_TFIDT(categories_above_10, counts_above_10)

                
        if max(filtered_counts_above_10) > unseen_category.shape[0]/2:
            set_unseen_category = 1
        elif max(filtered_counts_above_10) > sum(counts_above_10)/2:
            set_unseen_category = 2
        else:
            set_unseen_category = int(len(filtered_categories_above_10)/2)
        
        print(f'Categories above 10 = {filtered_categories_above_10}')
        print(f'Counts above 10 = {filtered_counts_above_10}')
        # unique_unseen_category = list(set(unseen_category))
        if not osp.exists(args.dataset+str(args.known_class)+'_categories_above_10.npy'):
            np.save(args.dataset+str(args.known_class)+'_categories_above_10.npy', filtered_categories_above_10)
        
    else:
        if len(g.label_names)-len(args.known_class) == 2:
            selected_unseen_indices = np.load(args.dataset+'_selected_unseen_indices.npy')
        else:
            selected_unseen_indices1 = np.load(args.dataset+'_selected_unseen_indices.npy')
            old_selected_unseen_indices1 = np.intersect1d(test_indices, selected_unseen_indices1)

            selected_unseen_indices2 = np.load(args.dataset+str(args.known_class)+'_selected_unseen_indices.npy') 
            selected_unseen_indices = np.concatenate([old_selected_unseen_indices1, selected_unseen_indices2])


    globalacc = 0
    save_path = 'results/' + str(args.model_type) + '/' + str(args.dataset) + '/' + str(args.seed) + '/' + str(args.fine_method)
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    ## fine-classifier
    test_close_acc, test_open_acc, test_overall_acc, predicted_labels = finetune_proser(59) 
    
    # ood-classifier   
    # predicted_labels = torch.load(os.path.join(save_path, 'predicted_labels.pth'))
    test_predicted_labels = predicted_labels[test_indices]
    indices_unseen = torch.where(test_predicted_labels == len(args.known_class))[0]
    test_indices_unseen = test_indices[indices_unseen.cpu().numpy()]
    
    if args.ood_classifier == "kmeans": 
        kmeans = KMeans(n_clusters=set_unseen_category, random_state=args.seed)
        x = g.x[test_indices_unseen].cpu().detach().numpy()
        kmeans.fit(x)
        cluster_labels = kmeans.labels_
    
    elif args.ood_classifier == "kmedoids":
        kmedoids = KMedoids(n_clusters=set_unseen_category)
        x = g.x[test_indices_unseen].cpu().detach().numpy()
        kmedoids.fit(x)
        cluster_labels = kmedoids.labels_
        
    elif args.ood_classifier == "llm_train":
        if args.dataset == 'cora':
            filtered_categories_above_10 = np.array(["Computer architecture", "Inductive logic programming", "Case based reasoning"])
            encoder.get_model()
            filtered_categories_above_10_embeddings = encoder.encode(filtered_categories_above_10) 
            original_categories_embeddings = encoder.encode(g.label_fullnames)

            similarity_matrix = cosine_similarity(filtered_categories_above_10_embeddings.cpu().numpy(), original_categories_embeddings.cpu().numpy())
            for i in range(len(filtered_categories_above_10)):
                j = similarity_matrix[i].argmax()
                if similarity_matrix[i].max() > 0.8:
                    filtered_categories_above_10[i] = g.label_fullnames[j]
                print(f'Replace {filtered_categories_above_10[i]} to {g.label_names[j]}')
                
        filtered_categories_above_10 = replace_category_by_annotation(filtered_categories_above_10.tolist(), g.label_fullnames)
    
        ood_prompts = generate_ood_prompts(g, args, predicted_category=filtered_categories_above_10, test_indices_unseen=test_indices_unseen)
        input_filename = "result_1v1/async_input_{}_model_{}_temperature_{}_n_{}_ood_input.json".format(args.dataset, args.llm_model_method, args.temperature, 1)
        output_filename = "result_1v1/async_input_{}_model_{}_temperature_{}_n_{}_ood_output.json".format(args.dataset, args.llm_model_method, args.temperature, 1)
   
        outputs = efficient_openai_text_ood(ood_prompts, input_filename, output_filename, model_name=args.llm_model_method, api_key=args.key, temperature=0.1, n=1)
        answer_list, confidence_list = get_result_from_ood(outputs)  
    elif args.ood_classifier in ['llama2_ood', 'llama3_ood']:
        if args.dataset == 'cora':
            filtered_categories_above_10 = np.array(["Computer architecture", "Inductive logic programming", "Case based reasoning"])
            encoder.get_model()
            filtered_categories_above_10_embeddings = encoder.encode(filtered_categories_above_10) 
            original_categories_embeddings = encoder.encode(g.label_fullnames)

            similarity_matrix = cosine_similarity(filtered_categories_above_10_embeddings.cpu().numpy(), original_categories_embeddings.cpu().numpy())
            for i in range(len(filtered_categories_above_10)):
                j = similarity_matrix[i].argmax()
                if similarity_matrix[i].max() > 0.8:
                    filtered_categories_above_10[i] = g.label_fullnames[j]
                print(f'Replace {filtered_categories_above_10[i]} to {g.label_names[j]}')
                
        filtered_categories_above_10 = replace_category_by_annotation(filtered_categories_above_10.tolist(), g.label_fullnames)
        use_prompt = 'zero_prompt2'   
        if args.ood_classifier == 'llama2_ood':
            model_dir = "/data/projects/punim1970/iclr2025/LLMGNN/Meta-Llama-2-7B"
        elif args.ood_classifier == 'llama3_ood':
            model_dir = "/data/projects/punim1970/iclr2025/LLMGNN/Meta-Llama-3-8B"   
        if not osp.exists(args.dataset+"_"+args.ood_classifier+"_answer_list_"+use_prompt+".npy"):   
            answer_list, confidence_list = llm_llama_ood(model_dir, g, args, predicted_category=filtered_categories_above_10, unseen_indices=unseen_indices, test_indices=test_indices_unseen)
            np.save(args.dataset+"_"+args.ood_classifier+"_answer_list_"+use_prompt+".npy", np.array(answer_list))
            np.save(args.dataset+"_"+args.ood_classifier+"_confidence_list_"+use_prompt+".npy", np.array(confidence_list))
        else:
            answer_list = np.load(args.dataset+"_"+args.ood_classifier+"_answer_list_"+use_prompt+".npy")
            confidence_list = np.load(args.dataset+"_"+args.ood_classifier+"_confidence_list_"+use_prompt+".npy")
     
    
    if args.dataset in ['cora', 'citeseer', 'wikics', 'dblp']:
        ood_labels = g.original_y[test_indices_unseen]
        ood_categories = [g.category_names[i] for i in test_indices_unseen]
        # if args.dataset != 'dblp':
        #     ood_categories = [full_mapping[args.dataset][x] for x in ood_categories]
        ood_categories = [x.lower() for x in ood_categories]
        
    if args.ood_classifier in ["kmeans", "kmedoids"]:
        true_cluster_num = []
        predicted_cluster_num = []        
        for i in range(len(unseen_label)):
            true_cluster_num.append((ood_labels == unseen_label[i]).sum())
            predicted_cluster_num.append((cluster_labels == i).sum())
            
        ood_accuracy = (min(min(true_cluster_num), min(predicted_cluster_num)) + min(max(true_cluster_num), max(predicted_cluster_num))) / len(unseen_indices)
    elif args.ood_classifier in ["llm_train", "llama2_ood", "llama3_ood"]:
        
        ood_accuracy = [1 if answer_list[i].lower() == ood_categories[i] else 0 for i in range(len(answer_list))]
        ood_accuracy_basedon_prediction = sum(ood_accuracy) / len(ood_accuracy)
        ood_accuracy = sum(ood_accuracy) / test_openset_indices.shape[0]
        overall_id_ood_accuracy = (test_close_acc * test_closeset_indices.shape[0] + ood_accuracy * test_openset_indices.shape[0])/test_indices.shape[0]
        
    else:
        ood_accuracy = 0
    
    # print(f'OOD accuracy = {ood_accuracy}, Overall ID OOD accuracy = {overall_id_ood_accuracy}, OOD accuracy based on prediction = {ood_accuracy_basedon_prediction}')
    print('done')    
    
    data = [args.model_type, args.dataset, args.lr, args.seed, "ID_class"+str(len(args.known_class)), args.fine_method, test_close_acc, test_open_acc, test_overall_acc]
    with open(f'main_results/{args.dataset}.csv', 'a', encoding='utf-8', newline='') as file_obj:
        if args.lamda != 1:
            data.append("lamda"+str(args.lamda))
        data.append(args.llm_name)
        data.append(args.mixup_num)
        writer = csv.writer(file_obj)
        writer.writerow(data) 
    
