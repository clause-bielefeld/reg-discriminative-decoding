# iterate through splits ('val' 'testA' 'testB')
for split in 'val' 'testA' 'testB'
do
  python decode_captions_baseline.py --refcoco_path $refcocoplus_path  --refcoco_model $refcocoplus_model --refcoco_vocab $refcocoplus_vocab --out_dir $refcocoplus_out_dir --image_dir $image_dir --split $split --beam_width $beam_width --max_len $max_len
done
