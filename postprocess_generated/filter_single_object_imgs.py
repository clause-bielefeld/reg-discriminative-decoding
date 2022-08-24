import pickle
import json
import os
import argparse
import pandas as pd
from numpy import logical_not


def prepare_ann_ids(args):

    def overview(dset, df, ann_ids):
        print(f'{dset}:')
        for split in list(pd.unique(df.split)):
            split_df = df.loc[df.split == split]
            n_imgs_split = len(split_df.groupby('image_id').first())
            split_len = len(split_df)
            sorted_out_df = split_df.loc[split_df.ann_id.isin(ann_ids)]
            sorted_out = len(sorted_out_df)
            n_imgs_sorted_out = len(sorted_out_df.groupby('image_id').first())
            left_df = split_df.loc[logical_not(split_df.ann_id.isin(ann_ids))]
            left = len(left_df)
            n_imgs_left = len(left_df.groupby('image_id').first())
            print(f"""Split {split}:
            total (entries, imgs): {split_len}, {n_imgs_split}
            sorted out (entries, imgs): {sorted_out}, {n_imgs_sorted_out}
            left (entries, imgs): {left}, {n_imgs_left}
            """)

    refcoco_path = args.refcoco_anns
    refcocoplus_path = args.refcocoplus_anns
    refcocog_path = args.refcocog_anns

    single_object_anns = dict()

    print('prepare ann ids')

    with open(refcoco_path, 'rb') as f:
        data = pickle.load(f)
    refcoco_df = pd.DataFrame(data)
    anns_per_image = refcoco_df.groupby('image_id').size()
    single_object_imgs = list(anns_per_image.loc[anns_per_image == 1].index)
    refcoco_single_object_anns = refcoco_df.loc[refcoco_df['image_id'].isin(single_object_imgs)].ann_id.to_list()
    overview('RefCOCO', refcoco_df, refcoco_single_object_anns)
    single_object_anns['refcoco'] = refcoco_single_object_anns

    with open(refcocoplus_path, 'rb') as f:
        data = pickle.load(f)
    refcocoplus_df = pd.DataFrame(data)
    anns_per_image = refcocoplus_df.groupby('image_id').size()
    single_object_imgs = list(anns_per_image.loc[anns_per_image == 1].index)
    refcocoplus_single_object_anns = refcocoplus_df.loc[refcocoplus_df['image_id'].isin(single_object_imgs)].ann_id.to_list()
    overview('RefCOCO+', refcocoplus_df, refcocoplus_single_object_anns)
    single_object_anns['refcocoplus'] = refcocoplus_single_object_anns

    with open(refcocog_path, 'rb') as f:
        data = pickle.load(f)
    refcocog_df = pd.DataFrame(data)
    anns_per_image = refcocog_df.groupby('image_id').size()
    single_object_imgs = list(anns_per_image.loc[anns_per_image == 1].index)
    refcocog_single_object_anns = refcocog_df.loc[refcocog_df['image_id'].isin(single_object_imgs)].ann_id.to_list()
    overview('RefCOCOg', refcocog_df, refcocog_single_object_anns)
    single_object_anns['refcocog'] = refcocog_single_object_anns

    return single_object_anns


def dset_split_from_fname(fname):
    """retrieve dataset and split informations from caption filename"""

    return fname.split('_')[0], fname.split('_')[1]


def filter_ann_file(args, single_object_anns):

    caps_file = os.path.split(args.caps_file)[-1]
    dataset_, split_ = dset_split_from_fname(caps_file)
    dataset_ = dataset_.replace('+', 'plus')

    split = split_ if not args.split else args.split
    out_file = args.caps_file.replace('.json', '_filtered.json') if not args.out_file else args.out_file

    print('caps file:', caps_file)
    print('dataset:', dataset_)
    print('split:', split)
    print('out file:', out_file)

    print('loading data...')

    # load caps file
    with open(args.caps_file) as f:
        caps = json.load(f)

    print('filter caps...')

    caps = [c for c in caps if c['target'] not in single_object_anns[dataset_]]
    assert 0 not in map(len, [c['distractors'] for c in caps])
    # write results to output file
    print('writing to file...')

    with open(out_file, 'w') as f:
        json.dump(caps, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--caps_file',
                        help='file (.json) containing model captions')
    parser.add_argument('--caps_dir',
                        default=None,
                        help='directory containing model caption files')
    parser.add_argument('--refcoco_anns',
                        # path to refcoco/refs(unc).p'
    )
    parser.add_argument('--refcocoplus_anns',
                        # path to refcoco+/refs(unc).p'
    )
    parser.add_argument('--refcocog_anns',
                        # path to refcocog/refs(umd).p'
    )

    parser.add_argument('--split',
                        default=None,
                        help='the current split (determined automatically if not specified)')
    parser.add_argument('--out_file',
                        default=None,
                        help='output file for results (determined automatically if not specified)')

    args = parser.parse_args()

    single_object_anns = prepare_ann_ids(args)

    if args.caps_dir:
        files = os.listdir(args.caps_dir)
        full_files = [os.path.join(args.caps_dir, f) for f in files]
        for f in full_files:
            assert os.path.isfile(f), 'file {} not found'.format(f)
            args.caps_file = f
            # create the file!
            filter_ann_file(args, single_object_anns)
    else:
        # create the file!
        filter_ann_file(args, single_object_anns)
