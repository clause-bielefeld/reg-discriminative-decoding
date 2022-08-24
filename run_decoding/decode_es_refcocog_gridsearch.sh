i=1
max_its=6

# iterate through splits ('val' 'testA' 'testB')
for SPLIT in 'test'
do

  # iterate through lambda values (0.3, 0.5 and 0.7)
  for LAMBDA in 0.1 0.2 0.4 0.6 0.8 0.9
  do

      # display current iteration
      printf "\n\n#############\n"
      printf "Iteration $i/$max_its"
      printf "\n#############\n\n"

      # execute python file with the arguments defined above
      # and the current values for --coco_cluster and --lambda_
      python decode_captions_es.py --refcoco_path $refcoco_path  --refcoco_model $refcoco_model --refcoco_vocab $refcoco_vocab --out_dir $refcoco_out_dir --image_dir $image_dir --split $SPLIT --lambda_ $LAMBDA --beam_width $beam_width --max_len $max_len

      # advance iteration counter
      i=$(($i+1))
  done

done
