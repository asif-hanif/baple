#!/bin/bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


DATASET_ROOT=/data-nvme/asif.hanif/datasets/med-datasets/
MODEL_ROOT=/data-nvme/asif.hanif/pre-trained-models/med-vlms/


MODEL=$1     # model name
DATASET=$2   # dataset name


if [[ $MODEL == "clip" || $MODEL == "plip" $MODEL == "quiltnet" ]] ; then
    python trainers/zeroshot/clip_plip_quiltnet.py --dataset_root $DATASET_ROOT --dataset_name $DATASET --model_root $MODEL_ROOT --model_name $MODEL
elif [[ $MODEL == "medclip" ]] ; then
    python trainers/zeroshot/medclip.py --dataset_root $DATASET_ROOT --dataset_name $DATASET --model_root $MODEL_ROOT --model_name $MODEL
elif [[ $MODEL == "biomedclip" ]] ; then
    python trainers/zeroshot/biomedclip.py --dataset_root $DATASET_ROOT --dataset_name $DATASET --model_root $MODEL_ROOT --model_name $MODEL
else
    echo "MODEL=$MODEL not supported."
    exit 1
fi


# if [[ $MODEL == "clip" ]] ; then
#     python trainers/zeroshot/clip_plip_quiltnet.py --dataset_root $DATASET_ROOT --dataset_name $DATASET --model_root $MODEL_ROOT --model_name "clip"
# elif [[ $MODEL == "plip" ]] ; then
#     python trainers/zeroshot/clip_plip_quiltnet.py --dataset_root $DATASET_ROOT --dataset_name $DATASET --model_root $MODEL_ROOT --model_name "plip"
# elif [[ $MODEL == "quiltnet" ]] ; then
#     python trainers/zeroshot/clip_plip_quiltnet.py --dataset_root $DATASET_ROOT --dataset_name $DATASET --model_root $MODEL_ROOT --model_name "quiltnet"
# elif [[ $MODEL == "medclip" ]] ; then
#     python trainers/zeroshot/medclip.py --dataset_root $DATASET_ROOT --dataset_name $DATASET --model_root $MODEL_ROOT --model_name $MODEL
# elif [[ $MODEL == "biomedclip" ]] ; then
#     python trainers/zeroshot/biomedclip.py --dataset_root $DATASET_ROOT --dataset_name $DATASET --model_root $MODEL_ROOT --model_name $MODEL
# else
#     echo "MODEL=$MODEL not supported."
#     exit 1
# fi