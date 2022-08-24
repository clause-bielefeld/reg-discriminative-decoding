import pandas as pd
import os
import json
from random import shuffle, seed

seed(123)  # Make it reproducible


# ---- #
# COCO #
# ---- #


def get_coco_captions(caps_path, return_caps='all'):
    """
        get dataframe containing captions for coco train and val images
    """

    # load captions for train2014 images as dataframe
    with open(caps_path+'annotations/captions_train2014.json') as file:
        captions = json.load(file)
        train_captions = pd.DataFrame(captions['annotations']).set_index('id')
        train_captions['coco_split'] = 'train'

    # load captions for val2014 images as dataframe
    with open(caps_path+'annotations/captions_val2014.json') as file:
        captions = json.load(file)
        val_captions = pd.DataFrame(captions['annotations']).set_index('id')
        val_captions['coco_split'] = 'val'

    if return_caps == 'all':
        # merge train2014 and val2014 annotations into single dataframe
        captions = pd.concat([train_captions, val_captions])
        return captions
    elif return_caps == 'train':
        return train_captions
    elif return_caps == 'val':
        return val_captions


def get_karpathy_splits(splits_path, return_df=False):
    """
        get image ids for karpathy splits
    """

    # set filepath and partition names
    filepath = os.path.join(splits_path, 'dataset_coco.json')

    # load karpathy splits as dataframe
    with open(filepath) as file:
        splits = json.load(file)
        splits = pd.DataFrame(splits['images'])

    if return_df:
        return splits

    # partitions: ['test', 'restval', 'val', 'train']
    partitions = list(pd.unique(splits.split))

    image_ids = {}

    for part in partitions:
        # get coco image ids for partitions
        image_ids[part] = splits.loc[splits.split == part].cocoid.to_list()

    return image_ids


def coco_speaker_listener_splits(splits_path, verbose=False):
    """
        split Karpathy train + restval partitions in half for speaker/listener
    """

    # get Karpathy split image_ids
    img_ids = get_karpathy_splits(splits_path)

    # combine train + restval images
    train_imgs = img_ids['train'] + img_ids['restval']
    # shuffle
    shuffle(train_imgs)

    half_of_imgs = round(len(train_imgs) / 2)

    # split data in half for speaker / listener splits
    speaker_train = train_imgs[half_of_imgs:]
    listener_train = train_imgs[:half_of_imgs]

    # validate
    if False in (
            # all train images in either speaker_train or listener_train
            set(speaker_train).union(set(listener_train)) == set(train_imgs),
            # no intersection
            len(set(speaker_train).intersection(set(listener_train))) == 0
            ):
        raise Exception("Error for speaker / listener splits")

    # output dict
    ids = {
        'speaker_train': speaker_train,
        'listener_train': listener_train,
        'val': img_ids['val'],
        'test': img_ids['test']
    }

    if verbose:
        print('images in partitions:')
        for key in ids.keys():
            print(key, len(ids[key]))

    return ids


# ------- #
# RefCOCO #
# ------- #


def split_sentences(df):
    """
        split sentences in refcoco df
    """
    rows = []

    def coco_split(row):
        for split in ['train', 'val', 'test']:
            if split in row['file_name']:
                return split
        return None

    def unstack_sentences(row):
        nonlocal rows
        for i in row.sentences:
            rows.append({
                'sent_id': i['sent_id'],
                'ann_id': row['ann_id'],
                'caption': i['sent'],
                'ref_id': row['ref_id'],
                'refcoco_split': row['split'],
                'coco_split': coco_split(row)
            })

    df.apply(lambda x: unstack_sentences(x), axis=1)

    return pd.DataFrame(rows)


def get_refcoco_captions(path):
    """
        get DataFrame containing referring expressions from RefCOCO
    """
    filepath = os.path.join(path, 'instances.json')
    with open(filepath) as file:
        instances = json.load(file)
        instances = pd.DataFrame(instances['annotations']).set_index('id')

    filepath = os.path.join(path, 'refs(unc).p')
    captions = pd.read_pickle(filepath)
    captions = split_sentences(pd.DataFrame(captions))

    captions = pd.merge(
            captions, instances[['image_id', 'bbox']],
            left_on='ann_id', right_on='id'
        ).set_index('sent_id')

    return captions


def refcoco_splits(path, return_captions=True):
    """
        get image/caption ids for RefCOCO train/val/test splits
    """
    captions = get_refcoco_captions(path)

    # partitions: ['train', 'testB', 'testA', 'val']
    partitions = list(pd.unique(captions.refcoco_split))

    image_ids, caption_ids = {}, {}

    for part in partitions:
        image_ids[part] = list(captions.loc[captions.refcoco_split == part].image_id.unique())
        caption_ids[part] = captions.loc[captions.refcoco_split == part].index.to_list()

    ids = {'image_ids': image_ids, 'caption_ids': caption_ids}

    if return_captions:
        return (captions, ids)

    return ids


def filename_from_id(image_id, prefix='', file_ending='.jpg'):
    """
    get image filename from id: pad image ids with zeroes,
    add file prefix and file ending
    """
    padded_ids = str(image_id).rjust(12, '0')
    filename = prefix + padded_ids + file_ending

    return (filename)


def get_refcocoplus_captions(path):
    return get_refcoco_captions(path=path)


def refcocoplus_splits(path, return_captions=True):
    return refcoco_splits(path=path,return_captions=return_captions)
