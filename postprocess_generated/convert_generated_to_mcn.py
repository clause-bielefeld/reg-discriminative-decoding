import pandas as pd
import os
import json
import pickle
import re
import argparse
from os.path import join


def get_data(ann_file, instances_file):
    """
    load refcoco / refcoco+ / refcocog
    and return a single DataFrame
    with referring expressions and instance annotations
    """

    # read annotation file
    with open(ann_file, 'rb') as f:
        anns = pd.DataFrame(pickle.load(f))

    # read instances file
    with open(instances_file) as f:
        instances = json.load(f)

    # get instance annotations and category descriptions
    instance_anns = pd.DataFrame(instances['annotations']).drop(columns = ['image_id', 'category_id'])
    instance_cats = pd.DataFrame(instances['categories'])

    # merge sentences and instance annotations
    merged = pd.merge(
        anns,
        instance_anns,
        left_on='ann_id',
        right_on='id'
    ).drop(columns='id')

    # merge sentences/instance annotations and category descriptions
    merged = pd.merge(
        merged,
        instance_cats,
        left_on='category_id',
        right_on='id'
    ).drop(columns='id')

    return merged


def bbox_process(bbox,cat,segement_id):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    box_info = " %d,%d,%d,%d,%d,%d" % (int(x_min), int(y_min), int(x_max), int(y_max), int(cat),int(segement_id))
    return box_info


def dset_split_from_fname(fname):
    """retrieve dataset and split informations from caption filename"""

    return fname.split('_')[0], fname.split('_')[1]


def create_ann_file(args):
    """create a text file containing information for ref ids and model captions"""

    caps_file = os.path.split(args.caps_file)[-1]
    dataset_, split_ = dset_split_from_fname(caps_file)

    dataset_ = dataset_.replace('+', 'plus')

    ann_file = vars(args)[dataset_+'_anns'] if not args.ann_file else args.ann_file
    instances_file = vars(args)[dataset_+'_instances'] if not args.instances_file else args.instances_file
    split = split_ if not args.split else args.split
    out_file = args.caps_file.replace('.json', '.txt') if not args.out_file else args.out_file

    print('\n')
    print('caps file:', caps_file)
    print('dataset:', dataset_)
    print('split:', split)
    print('annotation file:', ann_file)
    print('instances file:', instances_file)
    print('out file:', out_file)

    print('loading data...')

    # load annotations
    ref_data = get_data(ann_file, instances_file)

    # load caps file
    with open(args.caps_file) as f:
        caps = pd.DataFrame(json.load(f))

    # get ref ids from annotation file
    caps = pd.merge(
        caps,
        ref_data[['ann_id', 'ref_id']],
        left_on='target',
        right_on='ann_id'
    ).set_index('ref_id')

    # get all ref ids
    ref_ids = sorted(pd.unique(caps.index))

    # initialize global output string
    all_res = ''

    # iterate through ref ids in current split
    print('adding lines...')
    for i in ref_ids:
        # get corresponding entry
        ref = ref_data.loc[ref_data.ref_id == i].iloc[0]
        # get bbox, sentences, image filename, category_id
        bboxs = ref.bbox
        image_urls = re.sub(r'_\d+.jpg', '.jpg', ref.file_name)
        cat = ref.category_id
        # reformat bbox info
        box_info = bbox_process(bboxs, cat, i)

        # retrieve model caption for current id
        caption = caps.loc[i].caption

        # make output string for current id with img filename, bbox info and caption
        res = image_urls + box_info + ' ~ ' + caption


        # add all sentences for current id to output string
        #sentences = ref.sentences
        #for sentence in sentences:
        #    res += ' ~ '
        #    res += sentence['sent']

        # add current output string to global output string
        all_res += (res + '\n')

    # write results to output file
    print('writing to file...')

    with open(out_file, 'w') as f:
        f.write(all_res)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--caps_file',
                        help='file (.json) containing model captions')
    parser.add_argument('--caps_dir',
                        default=None,
                        help='directory containing model caption files')
    parser.add_argument('--refcoco_anns',
                        #path to refcoco/refs(unc).p'
    )
    parser.add_argument('--refcoco_instances',
                        #path to refcoco/instances.json'
    )
    parser.add_argument('--refcocoplus_anns',
                        #path to refcoco+/refs(unc).p'
    )
    parser.add_argument('--refcocoplus_instances',
                        #path to refcoco+/instances.json'
    )
    parser.add_argument('--refcocog_anns',
                        #path to refcocog/refs(umd).p'
    )
    parser.add_argument('--refcocog_instances',
                        #path to refcocog/instances.json'
    )

    parser.add_argument('--ann_file',
                        default=None,
                        help='file containing human annotations (determined automatically if not specified)')
    parser.add_argument('--instances_file',
                        default=None,
                        help='file containing bounding boxes and category information (determined automatically if not specified)')
    parser.add_argument('--split',
                        default=None,
                        help='the current split (determined automatically if not specified)')
    parser.add_argument('--out_file',
                        default=None,
                        help='output file for results (determined automatically if not specified)')

    args = parser.parse_args()


    if args.caps_dir:
        files = os.listdir(args.caps_dir)
        full_files = [os.path.join(args.caps_dir, f) for f in files]
        for f in sorted(full_files):
            assert os.path.isfile(f), 'file {} not found'.format(f)
            args.caps_file = f
            # create the file!
            create_ann_file(args)
    else:
        # create the file!
        create_ann_file(args)
