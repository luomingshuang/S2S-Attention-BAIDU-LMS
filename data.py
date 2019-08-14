import torch
from torch.utils.data import Dataset, DataLoader
import re
from cvtransforms import *
import cv2
import numpy as np
import os
import re
import editdistance


class MyDataset(Dataset):

    def __init__(self, **kwargs):
        # img_root, annotation_root, txt_file    
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        with open(self.txt_file, 'r') as f:            
            data = [line.strip() for line in f.readlines()]        
        
        self.data = []
        for line in data:
            with open(os.path.join(self.annotation_root, line), 'r') as f:
                annotation_lines = [line.strip() for line in f.readlines()]
                img_files = annotation_lines[2:]            
                if(len(img_files) <= self.img_padding): 
                    self.data.append(line)
        
    def __getitem__(self, idx):
        line = self.data[idx]
        part = re.split('[\/_\.]', line)
        img_folder = os.path.join(self.img_root, 
            '_'.join(part[2:6]), part[6][1:])
        with open(os.path.join(self.annotation_root, line), 'r') as f:
            annotation_lines = [line.strip() for line in f.readlines()]
            hanzi = annotation_lines[0]
            pinyin = annotation_lines[1]
            img_files = annotation_lines[2:]
            img_files = list(filter(lambda file: os.path.exists(os.path.join(img_folder, file)), img_files))
        images = [cv2.imread(os.path.join(img_folder, file)) for file in img_files]
        images = [cv2.resize(img, (96, 96)) for img in images]
        for i in range(self.img_padding-len(images)):
            images.append(np.zeros((96, 96, 3), dtype=np.uint8))
        images = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                      for img in images], axis=0) / 255.0
                      
        if(self.phase == 'train'):
            images = RandomCrop(images, (88, 88))
            images = ColorNormalize(images)
            images = HorizontalFlip(images)
            images = RandomDrop(images)
        elif self.phase == 'val' or self.phase == 'test':
            images = CenterCrop(images, (88, 88))
            images = ColorNormalize(images)
        # print(images[np.newaxis,...].shape)
        return {'encoder_tensor': torch.FloatTensor(images[np.newaxis,...]), 
            'decoder_tensor': torch.LongTensor(MyDataset.text2tensor(pinyin, self.text_padding, SOS=True))}

    def __len__(self):
        return len(self.data)
            
    @staticmethod
    def text2tensor(text, padding, SOS=False):
        # SOS: 1, EOS: 2, P: 0, <SPACE> 3, OTH: 4+x
        if(SOS):            
            tensor = [1]
        else:
            tensor = []
        for c in list(text):
            if(c == ' '): tensor.append(3)
            else: tensor.append(ord(c.lower())-ord('a')+4)
        tensor.append(2)
        for i in range(75-len(tensor)):
            tensor.append(0)
        return tensor
    
    @staticmethod
    def tensor2text(tensor):
        # (B, T)
        result = []
        n = tensor.size(0)
        T = tensor.size(1)
        for i in range(n):
            text = []
            for t in range(T):
                c = tensor[i,t]
                if(c == 2): break
                elif(c == 3): text.append(' ')
                elif(3 < c): text.append(chr(c-4+ord('a')))
            text = ''.join(text)
            result.append(text)
        return result
    
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer).mean()
                    
