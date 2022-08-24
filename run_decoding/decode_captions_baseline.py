import os
from os.path import join
import sys
import pickle
import torch
from torchvision import transforms
from tqdm.autonotebook import tqdm
import json
import argparse
from pprint import pprint

file_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.realpath(os.path.join(file_path, os.pardir))
os.chdir(dir_path)
sys.path.append(join(os.getcwd()))
sys.path.append(join(os.getcwd(), 'rsa'))
sys.path.append(join(os.getcwd(), 'model'))
#print(sys.path)

from decoding import img_from_file, prepare_img, greedy, beam_search
from predict_coco import get_img_dir
from model.build_vocab import Vocabulary
from model.adaptive_reg import Encoder2Decoder as REGEncoder2Decoder

from data_utils import refcoco_splits
from data_loader import get_reg_loader, RefCOCOClusters


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    dataset = os.path.split(args.refcoco_path)[-1]
    filename_template = '{dataset}_{split}_{method}_l-{lambda_}_r-{rationality}.json'

    # load vocab
    with open(args.refcoco_vocab, 'rb') as f:
        vocab = pickle.load(f)

    # define image transformation parameters
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])


    # load model
    model = REGEncoder2Decoder(256, len(vocab), 512)
    model.load_state_dict(torch.load(args.refcoco_model ,map_location='cpu' ))
    model.to(device)
    model.eval()

    beam_caps = list()
    greedy_caps = list()

    # prepare data loader
    ref_df = refcoco_splits(
        args.refcoco_path
    )[0]

    cluster_loader = RefCOCOClusters(
        split=[args.split],
        data_df=ref_df,
        image_dir=args.image_dir,
        vocab=vocab,
        decoding_level='word',
        transform=transform
    )

    pbar = tqdm(total=len(cluster_loader))

    # iterate through data loader
    for i, (sent_ids, ann_ids, images, positions, targets, filenames) in enumerate(cluster_loader):

        ann_ids = [int(i) for i in ann_ids]

        res_temp = {'target': ann_ids[0], 'distractors': ann_ids[1:]}

        imgs = [i.unsqueeze(0).to(device) for i in images]
        pos = [p.unsqueeze(0).to(device) for p in positions]

        img_t = imgs[0]
        pos_t = pos[0]

        # beam search
        prob, ids = beam_search(
            model, img_t,
            location_features=pos_t,
            beam_width=args.beam_width,
            max_len=args.max_len
        )
        words = [vocab.idx2word[ix] for ix in ids[:-1]]  # [:-1]: exclude <end>
        beam_res = {**res_temp}
        beam_res.update({
                'caption': ' '.join(words)
            })
        beam_caps.append(beam_res)

        # greedy
        ids, _, _ = greedy(
            model, img_t,
            location_features=pos_t,
            max_len=args.max_len
        )
        words = [vocab.idx2word[ix] for ix in ids[:-1]]  # [:-1]: exclude <end>
        greedy_res = {**res_temp}
        greedy_res.update({
                'caption': ' '.join(words)
            })
        greedy_caps.append(greedy_res)

        pbar.update()

    pbar.close()

    # save file

    # beam
    filename = filename_template.format(
        dataset=dataset,
        split=args.split,
        method='beam',
        lambda_='na',
        rationality='na',
    )
    filename = join(args.out_dir, filename)
    with open(filename, 'w') as f:
        json.dump(beam_caps, f)
        print('saved to', filename)

    # greedy
    filename = filename_template.format(
        dataset=dataset,
        split=args.split,
        method='greedy',
        lambda_='na',
        rationality='na',
    )
    filename = join(args.out_dir, filename)
    with open(filename, 'w') as f:
        json.dump(greedy_caps, f)
        print('saved to', filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', default='self', help='To make it runnable in jupyter')

    parser.add_argument('--refcoco_path', type=str)
    parser.add_argument('--refcoco_model', type=str)
    parser.add_argument('--refcoco_vocab', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--image_dir', type=str)

    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--separator', type=str, default=' ')

    args = parser.parse_args()
    args.separator = '' if args.separator in ['', 'char'] else ' '

    pprint(vars(args))

    main(args)
