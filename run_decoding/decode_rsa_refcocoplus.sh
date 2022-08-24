i=1
max_its=9

# iterate through splits ('val' 'testA' 'testB')
for SPLIT in 'val' 'testA' 'testB'
do

  # iterate through lambda values (0.3, 0.5 and 0.7)
  for RAT in 0.5 1.0 5.0
  do

      # display current iteration
      printf "\n\n#############\n"
      printf "Iteration $i/$max_its"
      printf "\n#############\n\n"

      # execute python file with the arguments defined above
      # and the current values for --coco_cluster and --lambda_
      python decode_captions_rsa.py --refcoco_path $refcocoplus_path  --refcoco_model $refcocoplus_model --refcoco_vocab $refcocoplus_vocab --out_dir $refcocoplus_out_dir --image_dir $image_dir --split $SPLIT --speaker_rat $RAT --beam_width $beam_width --max_len $max_len

      # advance iteration counter
      i=$(($i+1))
  done

done
