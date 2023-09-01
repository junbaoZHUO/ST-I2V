import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start_end):
    frames = []
    for i in start_end:
    # for i in range(start, start+num):
        DD = sorted(os.listdir(os.path.join(image_dir, vid)))
        # img = cv2.imread(os.path.join(image_dir, vid, str(i).zfill(5)+'.jpg'))[:, :, [2, 1, 0]]
        img = cv2.imread(os.path.join(image_dir, vid, DD[i]))[:, :, [2, 1, 0]]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(256,256))#,fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(image_list, split, root, mode, num_classes=101, dset = 'H'):
    dataset = []

    for val in open(image_list).readlines():
        if dset == 'H':
            AVI = val.split()[0].split('/')[-2]+'/'+val.split()[0].split('/')[-1][:-4]
            num_frames = len(os.listdir("/data/HMDB51-frame/"+AVI))
            if num_frames < 32: #num_used_frames=66
                print(val)
        elif dset == 'U':
            AVI = val.split()[0].split('/')[-2]+'/'+val.split()[0].split('/')[-1][:-4]
            num_frames = len(os.listdir("/data/UCF101_videos_frames/"+AVI))
            if num_frames < 32: #num_used_frames=66
                print(val)

        """
        if num_frames < 34: #num_used_frames=66
            continue
        """ # MM zhushidiaode

        """
        if not dset == 'E' and not dset=='S' and not dset == 'B':
            if num_frames < 34: #num_used_frames=66
                continue
        """
        # num_frames = 4
        NUM_=32
        if dset == 'E':
            num_frames = NUM_#16#16
            AVI = val.split()[0].split('/data/EH_features_img/EAD_image_dataset/')[1][:-4]
        elif dset == 'S':
            num_frames = NUM_##16
            AVI = val.split()[0].split('/data/Stanford40/JPEGImages/')[1][:-4]
            # AVI = val.split()[0].split('/data/EH_features_img/EAD_image_dataset/')[1].replace('/','_')[:-4]
        elif dset == 'B':
            num_frames = NUM_##16
            AVI = val.split()[0].split('/data//BU101/images/')[1][:-4]
        label = np.zeros((num_classes,num_frames), np.float32)

        for fr in range(0,num_frames,1):
            label[int(val.split()[1]), fr] = 1 # binary classification
        dataset.append((AVI, label, num_frames))
    
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, test=False, class_num=101, dset = 'H'):
        
        self.data = make_dataset(split_file, split, root, mode, class_num, dset)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.test = test
        self.dset = dset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        NUM_ = 16#32
        num_used_frame = NUM_#16#
        if self.dset == 'E':
            num_used_frame = NUM_#16#16#8
        vid, label, nf = self.data[index]
        """
        """
        if self.test:
            gap = int(float(nf-1)/num_used_frame)# - 1
            gap = max(gap,1)
            imgs = load_rgb_frames(self.root, vid, [int(i*gap)+1-1 for i in range(num_used_frame)])
        else:
            gap = int(float(nf-1)/num_used_frame)# - 1
            gap = max(gap,1)
            start_f = 1#random.randint(1,gap)#nf-num_used_frame-1)
            imgs = load_rgb_frames(self.root, vid, [int(i*gap)+start_f-1 for i in range(num_used_frame)])
            # start_f = random.randint(1,nf-num_used_frame-1)
            # imgs = load_rgb_frames(self.root, vid, range(start_f, start_f+num_used_frame))
            # 
        # start_f = random.randint(1,nf-num_used_frame-1)
        # imgs = load_rgb_frames(self.root, vid, range(start_f, start_f+num_used_frame))
        label = label[:, 0:num_used_frame]

        if len(self.transforms)>1:
            imgs_s = self.transforms[1](imgs.copy())
            imgs = self.transforms[0](imgs)
        else:
            imgs = self.transforms[0](imgs)

        if len(self.transforms)>1:
            return video_to_tensor(imgs), video_to_tensor(imgs_s), torch.from_numpy(label), index
        else:
            return video_to_tensor(imgs), torch.from_numpy(label), index

    def __len__(self):
        return len(self.data)



class Charades2(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, test=False, class_num=101, dset = 'H'):
        
        self.data = make_dataset(split_file, split, root, mode, class_num, dset)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.test = test
        self.dset = dset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf = self.data[index]
        NUM_ = 32
        num_used_frame = NUM_#16#
        if self.test:
            gap = int(float(nf-1)/num_used_frame) #@- 1
            gap = max(gap,1)
            imgs = load_rgb_frames(self.root, vid, [int(i*gap)+1-1 for i in range(num_used_frame)])
        else:
            gap = int(float(nf-1)/num_used_frame)# - 1
            gap = max(gap,1)
            start_f = random.randint(1,gap)#nf-num_used_frame-1)
            imgs = load_rgb_frames(self.root, vid, [int(i*gap)+start_f-1 for i in range(num_used_frame)])
            # start_f = random.randint(1,nf-num_used_frame-1)
            # imgs = load_rgb_frames(self.root, vid, range(start_f, start_f+num_used_frame))
            # 
        # start_f = random.randint(1,nf-num_used_frame-1)
        # imgs = load_rgb_frames(self.root, vid, range(start_f, start_f+num_used_frame))
        label = label[:, 0:num_used_frame]

        if len(self.transforms)>1:
            imgs_s = self.transforms[1](imgs.copy())
            imgs = self.transforms[0](imgs)
        else:
            imgs = self.transforms[0](imgs)

        if len(self.transforms)>1:
            return video_to_tensor(imgs), video_to_tensor(imgs_s), torch.from_numpy(label), index
        else:
            return video_to_tensor(imgs), torch.from_numpy(label), index

    def __len__(self):
        return len(self.data)

class Charades3(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, test=False, class_num=101, dset = 'H'):
        
        self.data = make_dataset(split_file, split, root, mode, class_num, dset)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.test = test
        self.dset = dset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        NUM_ = 16
        num_used_frame = NUM_#16#
        if self.dset == 'E':
            num_used_frame = NUM_#16#16#8
        vid, label, nf = self.data[index]
        """
        """
        imgs = load_rgb_frames2(self.root, vid, [0]*NUM_)
        label = label[:, 0:num_used_frame]

        if len(self.transforms)>1:
            imgs_s = self.transforms[1](imgs.copy())
            imgs = self.transforms[0](imgs)
        else:
            imgs = self.transforms[0](imgs)

        if len(self.transforms)>1:
            return video_to_tensor(imgs), video_to_tensor(imgs_s), torch.from_numpy(label), index
        else:
            return video_to_tensor(imgs), torch.from_numpy(label), index

    def __len__(self):
        return len(self.data)


def load_rgb_frames2(image_dir, vid, start_end):
    frames = []
    img = cv2.imread(os.path.join(image_dir, vid+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
    img = cv2.resize(img,dsize=(256,256))#,fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    for i in start_end:
        frames.append(img.copy())
    return np.asarray(frames, dtype=np.float32)


