# Import Modules
import os
import random
import pickle
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, preprocessed_path, phase='train', transform=None, caption_shuffle=True):
        with open(os.path.join(preprocessed_path, f'{phase}_processed.pkl'), 'rb') as f:
            data_dict = pickle.load(f)

        # Pre-setting
        self.phase = phase
        self.data_path = data_path
        self.transform = transform
        self.caption_shuffle = caption_shuffle
        
        # Post-setting
        self.data_dict = data_dict
        self.id_list = list(data_dict.keys())
        self.num_data = len(self.id_list)

    def __getitem__(self, index):
        ix = format(self.id_list[index], '012')
        # Image Open
        image = Image.open(os.path.join(self.data_path, f'{self.phase}2017/{ix}.jpg'))
        image = image.convert('RGB')
        # Image Augmentation
        if self.transform is not None:
            image = self.transform(image)
        # Caption Open
        if self.caption_shuffle:
            caption = random.choice(self.data_dict[int(ix)])
        else:
            caption = self.data_dict[int(ix)][0]

        return image, caption
    
    def __len__(self):
        return self.num_data

class PadCollate:
    def __init__(self, pad_index=0, dim=0):
        self.dim = dim
        self.pad_index = pad_index

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)

        def pack_sentence(sentences):
            sentences_len = max(map(lambda x: len(x), sentences))
            sentences = [pad_tensor(torch.LongTensor(seq), sentences_len, self.dim) for seq in sentences]
            sentences = torch.cat(sentences)
            sentences = sentences.view(-1, sentences_len)
            return sentences

        image, caption = zip(*batch)
        return torch.stack(image), pack_sentence(caption)

    def __call__(self, batch):
        return self.pad_collate(batch)