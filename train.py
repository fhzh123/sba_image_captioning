# Import Modules
# 1) PIP Moudle Import
import os
import time
import numpy as np
from math import ceil
from PIL import Image
from efficientnet_pytorch import EfficientNet

# 2) PyTorch Import
import torch
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# 3) Custom Module Import
from dataset import CustomDataset, PadCollate
from model.optimizer import Ralamb, WarmupLinearSchedule
from model.captioning_model import TransformerCaptioning
from utils import cal_loss, accuracy, save_checkpoint, str2bool

def training(args):
    #===================================#
    #===========Pre-Setting=============#
    #===================================#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = EfficientNet.get_image_size(args.efficientnet)
    efficientnet = EfficientNet.from_pretrained(args.efficientnet, advprop=True)
    efficientnet.eval()
    feature_dim, feature_size = efficientnet.extract_features(
        torch.FloatTensor(1, 3, image_size, image_size)).shape[1:3]

    args.smoothing_loss = str2bool(args.smoothing_loss)

    #===================================#
    #==========CustomDataset============#
    #===================================#
    transform_dict = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(15),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.RandomResizedCrop((image_size, image_size), 
                                        scale=(0.85, 1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        'valid': transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    dataset_dict = {
        'train': CustomDataset(args.coco_path, args.preprocessed_path, 
                               phase='train', transform=transform_dict['train']),
        'valid': CustomDataset(args.coco_path, args.preprocessed_path, 
                               phase='valid', transform=transform_dict['valid'])
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(), batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    }

    #===================================#
    #============Build Model============#
    #===================================#
    print("Build model")
    model = TransformerCaptioning(args.vocab_size, pad_idx=args.pad_idx, bos_idx=args.bos_idx, 
                eos_idx=args.eos_idx, max_len=args.max_len, feature_dim=feature_dim, feature_size=feature_size, 
                d_model=args.d_model, n_head=args.n_head, dim_feedforward=args.dim_feedforward,
                num_encoder_layer=args.num_encoder_layer, num_decoder_layer=args.num_decoder_layer, 
                dropout=args.dropout, embedding_dropout=args.embedding_dropout)
    model = model.to(device)
    model.extractor = efficientnet.to(device)
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

    optimizer = Ralamb(params=filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupLinearSchedule(optimizer, 
                    warmup_steps=ceil(len(dataloader_dict['train'])*args.num_epochs*0.1),
                    t_total=ceil(len(dataloader_dict['train'])*args.num_epochs))

    start_epoch = 0
    if args.resume:
        checkpoint_ = torch.load(args.checkpoint_path)
        start_epoch = checkpoint_['epoch']
        model.load_state_dict(checkpoint_['model'])
        optimizer.load_state_dict(checkpoint_['optimizer'])

    #===================================#
    #=============Training==============#
    #===================================#
    print("Training start")
    model.train()
    model.extractor.eval()
    model.extractor.requires_grad=False

    masks = {}
    for size in range(args.min_len, args.max_len+1):
        masks[size] = model.transformer.generate_square_subsequent_mask(size).to(device)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Model Fitting: [{epoch+1}/{args.num_epochs}]')
        start_time_e = time.time()
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                val_loss = 0
                model.eval()

            # Iterate over data
            for i, (img, cap) in enumerate(dataloader_dict[phase]):

                # Optimizer setting
                optimizer.zero_grad()

                # Iteration setting
                inputs = img.to(device, non_blocking=True)
                labels = cap.to(device, non_blocking=True)
                tgt_key_padding_mask = (labels == args.pad_idx)

                # Prediction
                with torch.set_grad_enabled(phase == 'train'):
                    predicted = model(img=inputs, dec_input_sentence=labels, 
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      tgt_mask=masks[labels.size(1)])

                    # Calculate loss
                    predicted_ = predicted.reshape(-1, predicted.size(-1))
                    targets_ = labels.contiguous().reshape(-1)
                    loss = cal_loss(predicted_, targets_, args.pad_idx, smoothing=args.smoothing_loss)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        if i % args.print_freq == 0:
                            print(f'[Iteration:{i+1}/{len(dataloader_dict["train"])}]', end=' ')
                            print(f'loss:{round(loss.item(), 2)} lr:{round(optimizer.param_groups[0]["lr"], 8)}', end=' ')
                            print(f'time:{round((time.time() - start_time_e) / 60, 2)}')

                    if phase == 'valid':
                        loss = F.cross_entropy(predicted_, targets_)
                        val_loss += loss.item()

        # Finishing iteration
        if phase == 'valid':
            val_loss /= len(dataloader_dict['valid'])
            print("[Epoch:%d] val_loss:%5.3f | spend_time:%5.2fmin"
                    % (epoch+1, val_loss, (time.time() - start_time_e) / 60))
            if (epoch+1)%10 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, epoch, args.save_path)