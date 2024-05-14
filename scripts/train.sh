#!/bin/sh

cfg=cfg/EAutoDet/pretrain_supernet-yolov5-mergenas.yaml
#cfg=cfg/coco_EAutoDet-yolov7-tiny_noAlpha.yaml
#cfg=cfg/coco_EAutoDet_noAlpha.yaml
#cfg=cfg/coco_retrain_EAutoDet.yaml
#cfg=cfg/cifar10_mergenas.yaml
#cfg=cfg/cifar10_retrain_mergenas.yaml

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
    START_CMD="python -u"
fi

#echo "Start?"
read -p "Start (Yes/No default Yes)? :" Answer
case $Answer in
    Yes|yes|y|Y|"")
#        echo "Start the process. Log file is saved to logs/${NAME}.log"
        CUDA_VISIBLE_DEVICES=$gpustr ${START_CMD} app/run_NN_engine.py \
		--cfg ${cfg} \
		> logs/EAutoDet/pretrain_supernet-yolov5-mergenas.log 2>&1 &
#                > logs/yolov7-tiny_225.log 2>&1 &
#		> logs/retrain_mergenas_bs64.log 2>& 1 &
#        	> logs/retrain_mergenas_noPermutation_noEdgeAlpha_noCutout_noActEachOp_SepConvInterReLU_alphaInit1e-3_actBeforeOp_gradClip5_noInitW_updateFreq1_withZeroOp.log 2>&1 &
        ;;
    No|no|N|n)
        echo "The process is killed!"
        ;;
esac

