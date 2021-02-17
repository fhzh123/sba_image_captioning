# Import modules
import os
import time
import argparse

# Import custom modules
from preprocessing import preprocessing
from train import training

def main(args):
    # Time setting
    total_start_time = time.time()

    if args.preprocessing:
        preprocessing(args)

    if args.training:
        training(args)
    
    # # NMT testing
    # if args.testing:
    #     testing(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='NMT argparser')
    # Process
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    # Path setting
    parser.add_argument('--preprocessed_path', default='./preprocessing', 
                        type=str, help='Preprocessed data path')
    parser.add_argument('--coco_path', default='/HDD/tnwls/data/coco/', 
                        type=str, help='COCO data path')
    parser.add_argument('--save_path', default='/HDD/kyohoon/sba_image_captioning')
    # Preprocessing setting
    parser.add_argument('--vocab_size', default=24000, type=int, help='Hanja vocabulary size; Default is 24000')
    parser.add_argument('--pad_idx', default=0, type=int, help='pad index')
    parser.add_argument('--bos_idx', default=1, type=int, help='index of bos token')
    parser.add_argument('--eos_idx', default=2, type=int, help='index of eos token')
    parser.add_argument('--unk_idx', default=3, type=int, help='index of unk token')
    # Model Setting
    parser.add_argument('--efficientnet', default='efficientnet-b2', type=str, 
                        help='version efficientnet model for image feature extractor')
    parser.add_argument('--min_len', default=4, type=int, help='min caption length')
    parser.add_argument('--max_len', default=300, type=int, help='max caption length')
    parser.add_argument('--d_model', default=768, type=int, help='model dimension')
    parser.add_argument('--n_head', default=8, type=int, help='number of head in self-attention')
    parser.add_argument('--dim_feedforward', default=3072, type=int, help='dimension of feedforward net')
    parser.add_argument('--num_encoder_layer', default=4, type=int, help='number of encoder layer')
    parser.add_argument('--num_decoder_layer', default=8, type=int, help='number of decoder layer')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout ratio')
    parser.add_argument('--embedding_dropout', default=0.1, type=float, help='embedding dropout ratio')
    # Training Setting
    parser.add_argument('--smoothing_loss', default='True', type=str, help='smoothing loss')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--num_epochs', default=80, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size per each worker')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for training dataloader')
    # ETC
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--print_freq', default=500, type=int)
    args = parser.parse_args()

    main(args)