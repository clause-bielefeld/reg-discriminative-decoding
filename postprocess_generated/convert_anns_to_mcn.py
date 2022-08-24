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

    ann_file = vars(args)[args.dataset.replace('+','plus') + '_anns'] if not args.ann_file else args.ann_file
    instances_file = vars(args)[args.dataset.replace('+','plus') + '_instances'] if not args.instances_file else args.instances_file
    split = args.split
    out_file = args.out_file

    #print('caps file:', caps_file)
    #print('dataset:', dataset_)
    print('split:', split)
    print('annotation file:', ann_file)
    print('instances file:', instances_file, '\n')
    print('out file:', out_file)

    print('loading data...')

    # load annotations
    ref_data = get_data(ann_file, instances_file)

    # get all ref ids
    ref_ids = sorted(pd.unique(ref_data.loc[ref_data.split == split].ref_id))

    # initialize global output string
    all_res = ''

    ref_data['sent'] = ref_data.sentences.map(lambda x: x[0]['sent'])
    sents = ref_data.set_index('ref_id')['sent']

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
        caption = sents.loc[i]

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

    out_file = './'+args.dataset+'_'+args.split+'_first_annotation.txt' if not args.out_file else args.out_file

    with open(out_file, 'w') as f:
        f.write(all_res)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

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

    parser.add_argument('--dataset',
                        default='refcoco')

    parser.add_argument('--ann_file',
                        default=None,
                        help='file containing human annotations (determined automatically if not specified)')
    parser.add_argument('--instances_file',
                        default=None,
                        help='file containing bounding boxes and category information (determined automatically if not specified)')
    parser.add_argument('--split',
                        default='val',
                        help='the current split (determined automatically if not specified)')
    parser.add_argument('--out_file',
                        default=None,
                        help='output file for results (determined automatically if not specified)')

    args = parser.parse_args()

    # create the file!
    create_ann_file(args)
