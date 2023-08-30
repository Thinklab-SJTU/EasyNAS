#!/bin/sh

cfg=cfg/validate/coco_EAutoDet.yaml

gpu=( $@ )
gpu_num=$#
gpustr=${gpu[0]}
for i in ${gpu[@]:1:$gpu_num-1};do
gpustr=$gpustr','$i;
done


echo "Use GPU: $gpustr"
echo "Configuration: ${cfg}"

if [ ${gpu_num} -gt 1 ]; then
#    START_CMD="torchrun --nnodes=1 --node_rank=0 --nproc_per_node=${gpu_num}"
    START_CMD="python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=${gpu_num} --use_env --master_port 9527"
else
    START_CMD="python"
fi

#echo "Start?"
read -p "Start (Yes/No default Yes)? :" Answer
case $Answer in
    Yes|yes|y|Y|"")
#        echo "Start the process. Log file is saved to logs/${NAME}.log"
        CUDA_VISIBLE_DEVICES=$gpustr ${START_CMD} app/train.py \
		--mode validate \
		--cfg ${cfg} \
#		> logs/validate.log 2>&1 &
        ;;
    No|no|N|n)
        echo "The process is killed!"
        ;;
esac


