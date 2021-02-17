# Import Modules
import os
import json
import pickle
import sentencepiece as spm

from PIL import Image
from glob import glob

def preprocessing(args):

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Path Setting
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # 2) JSON Open
    with open(os.path.join(args.coco_path, 'annotations/captions_train2017.json'), 'r') as f:
        train_captions = json.load(f)
    with open(os.path.join(args.coco_path, 'annotations/captions_valid2017.json'), 'r') as f:
        valid_captions = json.load(f)

    train_id_list, valid_id_list = list(), list()
    train_caption_list, valid_caption_list = list(), list()

    for annot in train_captions['annotations']:
        train_id_list.append(annot['image_id'])
        train_caption_list.append(annot['caption'].lower())

    for annot in valid_captions['annotations']:
        valid_id_list.append(annot['image_id'])
        valid_caption_list.append(annot['caption'].lower())


    #===================================#
    #===========SentencePiece===========#
    #===================================#

    # 1) Make Text to Train Vocabulary
    with open(f'{args.save_path}/text.txt', 'w') as f:
        for text in train_caption_list:
            f.write(f'{text}\n')

    # 2) SentencePiece Model Training
    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.save_path}/text.txt --model_prefix={args.save_path}/m_text --model_type=bpe '
        f'--vocab_size={args.vocab_size} --character_coverage=0.995 --split_by_whitespace=true '
        f'--pad_id={args.pad_idx} --unk_id={args.unk_idx} --bos_id={args.bos_idx} --eos_id={args.eos_idx}')

    # 3) Vocabulary Setting
    vocab = list()
    with open(f'{args.save_path}/m_text.vocab') as f:
        for line in f:
            vocab.append(line[:-1].split('\t')[0])
    word2id = {w: i for i, w in enumerate(vocab)}

    # 4) SentencePiece Model Load
    spm_ = spm.SentencePieceProcessor()
    spm_.Load(f"{args.save_path}/m_text.model")

    #===================================#
    #=============Encoding==============#
    #===================================#

    # 1) Parsing by SentencePiece model
    train_caption_indices = [[args.bos_idx] + spm_.EncodeAsIds(annot) + [args.eos_idx] for annot in train_caption_list]
    valid_caption_indices = [[args.bos_idx] + spm_.EncodeAsIds(annot) + [args.eos_idx] for annot in valid_caption_list]

    # 2) Train, Valid Encoding
    train_dict, valid_dict, test_dict = dict(), dict(), dict()

    for i in range(len(train_caption_indices)):
        try:
            train_dict[train_id_list[i]].append(train_caption_indices[i])
        except KeyError:
            train_dict[train_id_list[i]] = list()
            train_dict[train_id_list[i]].append(train_caption_indices[i])

    for i in range(len(valid_caption_indices)):
        try:
            valid_dict[valid_id_list[i]].append(valid_caption_indices[i])
        except KeyError:
            valid_dict[valid_id_list[i]] = list()
            valid_dict[valid_id_list[i]].append(valid_caption_indices[i])

    # 3) Saving
    with open(os.path.join(args.save_path, 'train_processed.pkl'), 'wb') as f:
        pickle.dump(train_dict, f)
    with open(os.path.join(args.save_path, 'valid_processed.pkl'), 'wb') as f:
        pickle.dump(valid_dict, f)