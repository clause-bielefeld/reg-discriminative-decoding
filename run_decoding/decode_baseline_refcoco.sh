# iterate through splits ('val' 'testA' 'testB')
for split in 'val' 'testA' 'testB'
do
  python decode_captions_baseline.py --refcoco_path $refcoco_path  --refcoco_model $refcoco_model --refcoco_vocab $refcoco_vocab --out_dir $refcoco_out_dir --image_dir $image_dir --split $split --beam_width $beam_width --max_len $max_len
done
