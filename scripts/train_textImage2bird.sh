MODEL='supervised'
# dataset details
CLASS='birds'
NC=256
NE=8
LOAD_SIZE=286
FINE_SIZE=256
INPUT_NC=1
NITER=150
NITER_DECAY=150

# training
GPU_ID=0
DISPLAY_ID=$((GPU_ID*10+5))
NAME=${MODEL}_image_${CLASS}
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --dataroot ../datasets/${CLASS} \
  --display_id ${DISPLAY_ID} \
  --name ${NAME} \
  --model ${MODEL} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --nc ${NC} \
  --ne ${NE} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --display_port 8098\
  --batchSize 2 \
  --c_type 'image_text'
  
  
