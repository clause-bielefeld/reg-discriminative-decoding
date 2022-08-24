# iterate through splits ('val' 'testA' 'testB')
for split in 'test'
do
  python decode_captions_baseline.py --refcoco_path $refcocog_path  --refcoco_model $refcocog_model --refcoco_vocab $refcocog_vocab --out_dir $refcocog_out_dir --image_dir $image_dir --split $split --beam_width $beam_width --max_len $max_len
done
