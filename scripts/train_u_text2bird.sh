MODEL='unsupervised'
# dataset details
CLASS='birds'
NC=256
LOAD_SIZE=286
FINE_SIZE=256
INPUT_NC=3
NITER=100
NITER_DECAY=100

# training
GPU_ID=1
DISPLAY_ID=$((GPU_ID*7+5))
NAME=${MODEL}_${CLASS}
# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --dataroot ../datasets/${CLASS} \
  --display_id ${DISPLAY_ID} \
  --name ${NAME} \
  --model ${MODEL} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --nc ${NC} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --display_port 8099\
  --batchSize 8 \
  --ngf 32 \
  --c_type 'text'
  
  
