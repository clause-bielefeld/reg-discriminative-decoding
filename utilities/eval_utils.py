import os
import sys
import json
import string
import spacy
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords

dir_path = os.path.dirname(os.path.realpath(__file__))
base_dir = os.path.join(dir_path, os.pardir)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'rsa'))
sys.path.append(os.path.join(base_dir, 'model'))

from coco.pycocotools.coco import COCO
from coco.pycocoevalcap.eval import COCOEvalCap
from decoding import img_from_file
from predict_coco import get_img_dir

nlp = spacy.load("en_core_web_sm")


def extract_noun_chunks(caption, s_model=nlp):
    """
    extract all noun chunks in caption as list
    """
    doc = s_model(caption)
    return [chunk.text for chunk in doc.noun_chunks]


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.

    From: https://stackoverflow.com/a/312464/2899924
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def type_token_ratio(
        sentences, n=1000, nostop=False,
        stops=stopwords.words('english')
        ):
    """
    Compute average type-token ratio (normalized over n tokens)
    with a repeated sample of n words.

    https://github.com/evanmiltenburg/MeasureDiversity/blob/master/methods.py
    """
    all_words = [word for sentence in sentences for word in sentence.split()]
    if nostop:
        all_words = [word for word in all_words if word not in stops]
    ttrs = []
    if len(all_words) < n:
        print("Warning: not enough tokens!")
        return None
    for chunk in chunks(all_words, n):
        if len(chunk) == n:
            types = set(chunk)
            ttr = float(len(types))/n
            ttrs.append(ttr)
    final_ttr = float(sum(ttrs))/len(ttrs)
    return final_ttr


def dist_n(caps_list, n):

    tokens = [word_tokenize(c) for c in caps_list]
    n_grams = []
    for ts in tokens:
        n_grams += list(ngrams(ts, n))

    return (len(set(n_grams)) / len(n_grams))


def display_row(row, args):

    cols = sorted(row.keys().drop(['target', 'distractors']))
    target_img = img_from_file(
            get_img_dir(row.target, args.image_dir)
        ).resize((args.crop_size, args.crop_size))

    distractor_imgs = [
        img_from_file(
            get_img_dir(d, args.image_dir)
        ).resize((args.crop_size, args.crop_size))
        for d in row.distractors
    ]
    imgs = [target_img] + distractor_imgs

    print('target:', row.target)
    print('distractors:', row.distractors)

    print('\n')

    for c in cols:
        print('{}:'.format(c).ljust(15), '{}'.format(row[c]))

    print('\n')

    fig = plt.figure(figsize=(8, 8))
    for i in range(4):
        fig.add_subplot(2, 2, i+1)
        plt.imshow(imgs[i])
    plt.show()

    print('\n--------------------\n')


def get_coco_captions(path):
    """ get dataframe containing captions for coco train and val images """

    # load captions for train2014 images as dataframe
    with open(path+'annotations/captions_train2014.json') as file:
        captions = json.load(file)
        train_captions = pd.DataFrame(captions['annotations']).set_index('id')
        train_captions['coco_split'] = 'train'
    # load captions for val2014 images as dataframe
    with open(path+'annotations/captions_val2014.json') as file:
        captions = json.load(file)
        val_captions = pd.DataFrame(captions['annotations']).set_index('id')
        val_captions['coco_split'] = 'val'

    # merge train2014 and val2014 annotations into single dataframe
    captions = pd.concat([train_captions, val_captions])

    return captions


def coco_caption_splits(splits_path, captions_path):
    """
        get image_ids and caption_ids for karpathy
        train, val, test, and restval splits
    """

    # set filepath and partition names
    filepath = os.path.join(splits_path, 'dataset_coco.json')
    partitions = ['test', 'restval', 'val', 'train']

    # return error and link to karpathy splits if file is not found
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            "File {path} doesn't exist".format(path=filepath)
            )
    # load karpathy splits as dataframe
    with open(filepath) as file:
        splits = json.load(file)
        splits = pd.DataFrame(splits['images'])
    # create dict with image_ids for each partition
    image_ids = {
        part: splits.loc[splits.split == part].cocoid.to_list()
        for part in partitions
    }
    # load coco captions, create dict with caption_ids for each partition
    captions = get_coco_captions(path=captions_path)
    caption_ids = {
        part: captions.loc[
                captions.image_id.isin(image_ids[part])
            ].index.to_list()
        for part in partitions
    }

    ids = {'image_ids': image_ids, 'caption_ids': caption_ids}

    return (captions, ids)


def count_tokens(sent_list, vocab=None, stops=None):
    all_sents = ' '.join(sent_list)
    all_sents = all_sents.lower().translate(
        str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(all_sents)

    if vocab is not None:
        tokens = [t for t in tokens if t in vocab.idx2word.values()]
    if stops is not None:
        tokens = [t for t in tokens if t not in stops]

    return Counter(tokens)


class ModelEvaluator:
    # taken from here: 09-diversity-pragmatics/code/models/adaptiveg/utils.py
    '''To evaluate generated captions'''
    def __init__(self, caption_val_path):
        self.coco = COCO(caption_val_path)

    def compute_scores(self, res_file):
        '''Compute scores of evaluation metrics'''

        coco_res = self.coco.loadRes(res_file)

        coco_eval = COCOEvalCap(self.coco, coco_res)
        coco_eval.params['image_id'] = coco_res.getImgIds()
        coco_eval.evaluate()

        return coco_eval.eval
