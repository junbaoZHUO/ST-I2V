import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-droot', type=str)
parser.add_argument('-CLASS_NUM', type=int)
parser.add_argument('-dset', type=str)
parser.add_argument('-sset', type=str)
parser.add_argument('-method', type=str)
parser.add_argument('-trainfile', type=str)
parser.add_argument('-r1', type=float)
parser.add_argument('-r2', type=float)
parser.add_argument('-r3', type=float)
parser.add_argument('-th', type=float)
parser.add_argument('-lr', type=float)
parser.add_argument('-PRE', type=float)
parser.add_argument('-update', type=float, default=1)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from datasets import Charades as Dataset
from datasets import Charades2 as Dataset2
import random

from torch.autograd import Function


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)#/len(kernel_val)


def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1+1, batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss1 += kernels[s1, s2] + kernels[t1, t2]
            loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss2 -= kernels[s1, t2] + kernels[s2, t1]
            loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2


def JAN(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1 + 1, batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss1 += joint_kernels[s1, s2] + joint_kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss2 -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
       ctx.lambd = lambd
       return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambd), None



def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    # print(type(max_iter))
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def mlp_loss(features, ad_net, max_iter):
    ad_out = ad_net(features,max_iter=max_iter)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


def Entropy(input_,dd = None):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if dd!=None:
        entropy = torch.sum(entropy, dim=dd)
    else:
        entropy = torch.sum(entropy, dim=1)
    return entropy


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000

  def forward(self, x, max_iter,reverse=True):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, max_iter)
    x = x * 1.0
    if reverse:
        x = grad_reverse(x, coeff)
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2,'lr':0.001}]

def Hentropy(outputs, lamda=1):
    out_t1 = F.softmax(outputs)
    loss_ent = -lamda * torch.mean(out_t1 * (torch.log(out_t1 + 1e-5)))
    return loss_ent

def CDAN(input_list, ad_net, max_iter, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)),max_iter=max_iter)

    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)),max_iter=max_iter)
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
        target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


def MCC(outputs_target,args):
    outputs_target_temp = outputs_target / args
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    sample_confusion = torch.zeros(outputs_target.size(0),1).cuda()
    for idx in range(outputs_target.size(0)):
        cov_ma = target_softmax_out_temp[idx].view(1,-1).transpose(1,0).mm(target_softmax_out_temp[idx].view(1,-1))
        sample_confusion[idx] = torch.sum(cov_ma) - torch.trace(cov_ma)
    cov_matrix_t = target_softmax_out_temp.transpose(1,0).mm(target_softmax_out_temp)
    sum_cov = torch.sum(cov_matrix_t, dim=1)
    en_loss = Hentropy(sum_cov)
    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / outputs_target.size(1)
    return mcc_loss, sample_confusion, cov_matrix_t

def bnm(out_u, lamda, eta=1.0):
    softmax_u = nn.Softmax(dim=1)(out_u)
    _, s_u, _ = torch.svd(softmax_u)
    loss_adent = -lamda*torch.mean(s_u)
    return loss_adent

def fbnm(out_u, lamda, eta=1.0):
    softmax_tgt = nn.Softmax(dim=1)(out_u)
    list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_tgt,2),dim=0)), descending=True)
    loss = - torch.mean(list_svd[:min(softmax_tgt.shape[0],softmax_tgt.shape[1])])
    return loss

def entropy(output_target):
    softmax = nn.Softmax(dim=1)
    output = output_target
    output = softmax(output)
    en = -torch.sum((output*torch.log(output + 1e-8)), 1)
    return en#torch.mean(en)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



def run(init_lr=0.05, max_steps=20, mode='rgb', root='/data1/HMDB51-frame/', train_split='../targetlistname_My_H.txt', batch_size=16, save_model='I3D_RGB', CLASS_NUM = 13, dset='H', args=None):
    # setup dataset
    CLASS_NUM = CLASS_NUM
    dset = dset
    train_transforms = [transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])]#,

    test_transforms = [transforms.Compose([videotransforms.CenterCrop(224)])]

    dataset = Dataset('./sourcelistname_My_E.txt', 'training', root, mode, train_transforms, class_num=CLASS_NUM, dset=args.sset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last = True)
    tgt_dataset = Dataset(train_split, 'training', args.droot, mode, train_transforms, class_num=CLASS_NUM, dset=dset)
    tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)    

    val_dataset = Dataset2(train_split, 'testing', args.droot, mode, test_transforms, test=True,  class_num=CLASS_NUM, dset=dset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader, 'tgt':tgt_dataloader}
    datasets = {'train': dataset, 'val': val_dataset, 'tgt': tgt_dataset}

    
    # setup the model
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('./models/rgb_imagenet.pt'))
    i3d.replace_logits(CLASS_NUM)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    if 'CDAN' in args.method:
        adnet = AdversarialNetwork(1024*CLASS_NUM,1024).cuda()
    if 'DANN' in args.method:
        adnet = AdversarialNetwork(1024, 1024).cuda()

    lr = args.lr
    if 'CDAN' in args.method or 'DANN' in args.method:
        optimizer = optim.SGD(list(i3d.parameters())+list(adnet.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001) 
    else:
        optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001) # 0.0000001
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=0.1)


    num_steps_per_update = args.update # accum gradient
    steps = 0
    # train it
    FEAT = torch.zeros((dataset.__len__(), 384+384+128+128)).cuda()
    FEAT_TGT = torch.zeros((tgt_dataset.__len__(), 384+384+128+128)).cuda()
    LOSS = torch.zeros((40, dataset.__len__(), 1)).cuda()
    TARGET = torch.zeros((dataset.__len__(), CLASS_NUM)).cuda()
    PPP1 = torch.zeros((dataset.__len__())).long().cuda()
    PPP2 = torch.zeros((val_dataset.__len__())).long().cuda()
    EST = torch.zeros((40, dataset.__len__(), 1)).cuda()
    SELECT_IND = []

    PRE = args.PRE
    STRONG = False
    METHOD = args.method
    RATIO = 0.5
    PSEUDO = None
    targets2 = None

    while steps < max_steps:#for epoch in range(num_epochs):
        print ('Step {}/{}'.format(steps, max_steps))
        print ('-' * 10)

        FEAT = torch.zeros((dataset.__len__(), 384+384+128+128)).cuda()
        FEAT_TGT = torch.zeros((tgt_dataset.__len__(), 384+384+128+128)).cuda()
        # Each epoch has a training and validation phase
        if 'CDAN' in args.method:
            adnet.train()
        if 'DANN' in args.method:
            adnet.train()
        start_test = True
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            ACC=0
            LEN=0
            iter_tgt = iter(tgt_dataloader)
            len_it = len(iter_tgt)
            iter_tgt_num = 0 


            for data in dataloaders[phase]:
                iter_tgt_num +=1
                num_iter += 1
                # get the inputs
                if phase == 'train':
                    inputs, labels, index = data
                    if iter_tgt_num == len_it - 1:
                        iter_tgt_num = 1 
                        iter_tgt = iter(tgt_dataloader)
                    inputs_tgt, _, index_tgt = next(iter_tgt)#.next()
                else:
                    inputs, labels, index = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                if phase == 'train':
                    inputs_tgt = Variable(inputs_tgt.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                if phase == 'train':
                    per_frame_logits, feat_ = i3d(inputs)
                    per_frame_logits_tgt, ft_ = i3d(inputs_tgt)
                    FEAT__ = torch.cat((feat_.mean(dim=2).view(feat_.size(0),-1),ft_.mean(dim=2).view(ft_.size(0),-1)), dim=0)
                else:
                    with torch.no_grad():
                        per_frame_logits, feat_ = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
                if phase == 'train':
                    per_frame_logits_tgt = F.upsample(per_frame_logits_tgt, t, mode='linear')
                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                # compute localization loss
                tot_loc_loss =0#+= loc_loss#.data[0]

                # compute classification loss (with max-pooling along time B x C x T)
                if phase == 'train':

                    feat_ = feat_.mean(dim=2).view(feat_.size(0),-1)
                    feat_ = feat_ / (torch.norm(feat_, p=2, dim=1, keepdim=True) + 0.00001)
                    FEAT[index] = feat_ 


                    cls_loss = F.binary_cross_entropy_with_logits(torch.mean(per_frame_logits, dim=2), torch.max(labels, dim=2)[0], reduction='none').mean(dim=1).view(-1, 1) #* targets.view(-1,1)
                    tot_cls_loss += cls_loss.sum()#/(targets.sum()+0.000001)#.data[0]
                    LOSS[steps, index] = cls_loss.data.detach()
                    PPP1[index] = torch.max(torch.max(labels, dim=2)[0], dim=1)[1].data.detach()
                else:
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0], reduction='none').mean(dim=1).view(-1, 1) #* targets.view(-1,1)
                    feat_ = feat_.mean(dim=2).view(feat_.size(0),-1)
                    feat_ = feat_ / (torch.norm(feat_, p=2, dim=1, keepdim=True) + 0.00001)
                    FEAT_TGT[index] = feat_ 




                if phase == 'val':
                    ACC+= (torch.max(torch.mean(per_frame_logits, dim=2),dim=1)[1] == torch.max(torch.max(labels, dim=2)[0], dim=1)[1]).float().sum()
                    LEN+=torch.max(per_frame_logits, dim=2)[0].shape[0]
                    PPP2[index] = torch.max(torch.max(labels, dim=2)[0], dim=1)[1].data.detach()
                    if start_test:
                        all_output = F.softmax(torch.mean(per_frame_logits, dim=2),dim=1).float().cpu()
                        all_output1 = torch.mean(per_frame_logits, dim=2).float().cpu()
                        all_label = torch.max(torch.max(labels, dim=2)[0], dim=1)[1].float()
                        start_test = False
                    else:
                        all_output = torch.cat((all_output, F.softmax(torch.mean(per_frame_logits, dim=2),dim=1).float().cpu()), 0)
                        all_output1 = torch.cat((all_output1, torch.mean(per_frame_logits, dim=2).float().cpu()), 0)
                        all_label = torch.cat((all_label, torch.max(torch.max(labels, dim=2)[0], dim=1)[1].float()),0)
 

                if phase == 'train':
                    if steps < PRE: # PRE>20 BNM or Source Only ;  PRE < 20, BNM then Fix+BNM 
                        if METHOD == "Source":
                            loss = cls_loss.mean()/num_steps_per_update
                            loss += loc_loss
                        elif METHOD == "MCC":
                            mcc_loss, en_loss, _ = MCC(torch.mean(per_frame_logits_tgt, dim=2), 2.5)
                            loss = args.r2*mcc_loss/num_steps_per_update + cls_loss.mean()/num_steps_per_update
                            loss += loc_loss

                        elif args.method=='CDAN':
                            target_ = torch.cat((torch.mean(per_frame_logits,dim=2), torch.mean(per_frame_logits_tgt, dim=2)),dim=0)
                            softmax_target_ = nn.Softmax(dim=1)(target_)
                            entropy_ = Entropy(softmax_target_)
                            transfer_loss = CDAN([FEAT__,softmax_target_],adnet, 1000,entropy_, calc_coeff(num_iter,max_iter=1000),None)
                            loss = args.r2 * transfer_loss/num_steps_per_update + cls_loss.mean()/num_steps_per_update
                            loss += loc_loss
                        elif args.method=='DANN':
                            transfer_loss = mlp_loss(FEAT__,adnet, 1000)#,entropy_,network.calc_coeff(iter_num,max_iter=max_iter),None)
                            loss = args.r2*transfer_loss/num_steps_per_update + cls_loss.mean()/num_steps_per_update
                            loss += loc_loss
 
                        elif args.method=='DAN':
                            transfer_loss = DAN(torch.mean(per_frame_logits,dim=2), torch.mean(per_frame_logits_tgt, dim=2), kernel_mul=2.0, kernel_num=5, fix_sigma=None)
                            loss = args.r2*transfer_loss/num_steps_per_update + cls_loss.mean()/num_steps_per_update
                            loss += loc_loss
                        else:
                            entLoss = fbnm(torch.mean(per_frame_logits_tgt, dim=2),1)
                            loss = args.r2*entLoss.mean()/num_steps_per_update + cls_loss.mean()/num_steps_per_update
                            loss += loc_loss
                    else:
                        if METHOD == "Ent":
                            entLoss = entropy(torch.max(per_frame_logits, dim=2)[0]).view(-1,1)*targets2.view(-1,1)
                            loss = (0.05*entLoss.sum()/(targets2.sum()+0.000001) + cls_loss.sum()/(targets.sum()+0.000001))/num_steps_per_update
                            loss += loc_loss

                        else:
                            if PSEUDO is None:
                                loss = cls_loss.sum()/targets.sum()/num_steps_per_update
                            else:
                                    
                                """
                                """
                                small_loss = F.cross_entropy(torch.mean(per_frame_logits_tgt, dim=2), PSEUDO[index_tgt], reduction='none').view(-1, 1).detach()
                                targets2 = small_loss * 0 
                                aa,bb= torch.sort(small_loss.view(-1))
                                for iii in bb[:int(bb.size(0)*args.r3)]:
                                    targets2[iii]=1
                                u_loss = F.cross_entropy(torch.mean(per_frame_logits_tgt, dim=2), PSEUDO[index_tgt], reduction='none').view(-1, 1) * targets2.cuda().view(-1,1)

                                if METHOD == "MCC":
                                    entLoss, en_loss, _ = MCC(torch.mean(per_frame_logits_tgt, dim=2), 2.5)
                                elif METHOD == "BNM":
                                    entLoss = fbnm(torch.mean(per_frame_logits_tgt, dim=2),1)
                                elif METHOD == "DANN":
                                    entLoss = mlp_loss(FEAT__,adnet, 1000)
                                elif METHOD == "CDAN":
                                    target_ = torch.cat((torch.mean(per_frame_logits,dim=2), torch.mean(per_frame_logits_tgt, dim=2)),dim=0)
                                    softmax_target_ = nn.Softmax(dim=1)(target_)
                                    entropy_ = Entropy(softmax_target_)
                                    entLoss = CDAN([FEAT__,softmax_target_],adnet, 1000,entropy_, calc_coeff(num_iter,max_iter=1000),None)
                                else :
                                    entLoss = DAN(torch.mean(per_frame_logits,dim=2), torch.mean(per_frame_logits_tgt, dim=2), kernel_mul=2.0, kernel_num=5, fix_sigma=None)

                                loss = (cls_loss.mean()+u_loss.sum()/(targets2.cuda().sum()+0.000001)*args.r1)/num_steps_per_update + args.r2*entLoss.mean()
                                loss += loc_loss
                    tot_loss += loss#.data[0]

                    loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    num_iter = 0
                    clip_gradient(optimizer, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if steps % 1 == 0:
                        print ('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(1*num_steps_per_update), tot_cls_loss/(1*num_steps_per_update), tot_loss/1))
                        # save model
                        # torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                print(ACC/LEN)
    
        steps += 1
        lr_sched.step()
        if phase == 'val':
            PROTO = torch.zeros((CLASS_NUM, 384+384+128+128)).cuda()
            CNT = torch.zeros((CLASS_NUM, 1)).cuda()
            for iii in range(tgt_dataset.__len__()):
                PROTO[torch.argmax(all_output[iii])] += FEAT_TGT[iii]
                CNT[torch.argmax(all_output[iii])] += 1
            AAA=open('./sourcelistname_My_E.txt').readlines()
            for iii in range(dataset.__len__()):
                PROTO[int(AAA[iii].split()[-1])] += FEAT[iii]
                CNT[int(AAA[iii].split()[-1])] += 1
            for iii in range(CLASS_NUM):
                PROTO[iii,:] = PROTO[iii,:]/CNT[iii]
            SIM = torch.matmul(FEAT_TGT, PROTO.t())
            PSEUDO = torch.max(SIM,dim=1)[1]
            print((PSEUDO==all_label).float().mean())





if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, save_model=args.save_model, CLASS_NUM=args.CLASS_NUM, dset=args.dset, train_split=args.trainfile, args=args)
