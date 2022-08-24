#!/bin/sh

data_root="/home/simeon/Dokumente/Datasets"  # data set location
# default: directory with COCO, refcoco, refcoco+ and refcocog subdirs
# (change paths below if necessary)
model_root="../data/model"  # location of the model
out_root="../data/model_expressions"  # dir for generated expressions

export beam_width="5"
export max_len="20"

export image_dir=`realpath "${data_root}/COCO/"`  # you might have to change this

export refcoco_path=`realpath "${data_root}/refcoco"`
export refcoco_model=`realpath "${model_root}/refcoco/best.pkl"`
export refcoco_vocab=`realpath "${model_root}/refcoco/refcoco_vocab.pkl"`
export refcoco_out_dir=`realpath "${out_root}/refcoco/"`

export refcocoplus_path=`realpath "${data_root}/refcoco+"`
export refcocoplus_model=`realpath "${model_root}/refcoco+/best.pkl"`
export refcocoplus_vocab=`realpath "${model_root}/refcoco+/refcocoplus_vocab.pkl"`
export refcocoplus_out_dir=`realpath "${out_root}/refcocoplus/"`

export refcocog_path=`realpath "${data_root}/refcocog"`
export refcocog_model=`realpath "${model_root}/refcocog/best.pkl"`
export refcocog_vocab=`realpath "${model_root}/refcocog/refcocog_vocab.pkl"`
export refcocog_out_dir=`realpath "${out_root}/refcocog/"`


# # Baseline

echo "Baseline RefCOCO"; bash decode_baseline_refcoco.sh  # Beam / Greedy  - RefCOCO
echo "Baseline RefCOCOg"; bash decode_baseline_refcocog.sh  # Beam / Greedy  - RefCOCOg
echo "Baseline RefCOCO+"; bash decode_baseline_refcocoplus.sh  # Beam / Greedy  - RefCOCO+

# # ES

echo "ES RefCOCO"; bash decode_es_refcoco.sh  # ES - RefCOCO
echo "ES RefCOCOg"; bash decode_es_refcocog.sh  # ES - RefCOCOg
echo "ES RefCOCO+"; bash decode_es_refcocoplus.sh  # ES - RefCOCO+

# RSA

echo "RSA RefCOCO"; bash decode_rsa_refcoco.sh  # RSA - RefCOCO
echo "RSA RefCOCOg"; bash decode_rsa_refcocog.sh  # RSA - RefCOCOg
echo "RSA RefCOCO+"; bash decode_rsa_refcocoplus.sh  # RSA - RefCOCO+
