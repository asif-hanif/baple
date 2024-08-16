#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd Dassl.pytorch && python setup.py develop && cd ..


DATASET_ROOT=/data-nvme/asif.hanif/datasets/med-datasets/
MODEL_ROOT=/data-nvme/asif.hanif/pre-trained-models/med-vlms/



MODEL=$1     # model name
DATASET=$2   # dataset name
CFG=$3       # config file
SHOTS=$4     # number of shots (1, 2, 4, 8, 16)
NCTX=16      # number of context tokens       [COOP]
CTP=end      # class token position (end)     [COOP]
CSC=False    # class-specific context (False) [COOP]



# TRAINER
if [[ $MODEL == "clip" || $MODEL == "plip" || $MODEL == "quiltnet" ]] ; then
    TRAINER=CoOp
elif
    [[ $MODEL == "medclip" ]] ; then
    TRAINER=CoOp_MedCLIP
elif [[ $MODEL == "biomedclip" ]] ; then
    TRAINER=CoOp_BioMedCLIP
else
    echo "MODEL=$MODEL not supported."
    exit 1
fi


## TARGET CLASSES
if [[ $DATASET == "pannuke" || $DATASET == "digestpath" || $DATASET == "wsss4luad" || $DATASET == "covid" ]] ; then
    TARGET_CLASSES=(0 1)
elif [[ $DATASET == "kather" ]] ; then
    TARGET_CLASSES=(0 1 2 3 4 5 6 7 8)
elif [[ $DATASET == "rsna18" ]] ; then
    TARGET_CLASSES=(0 1 2)
elif [[ $DATASET == "mimic" ]] ; then
    TARGET_CLASSES=(0 1 2 3 4)
else
    echo "DATASET=$DATASET not supported."
    exit 1
fi




for TARGET_CLASS in ${TARGET_CLASSES[@]}
    do
        printf "\n\n\n\n\n"
        echo "###############################################"
        echo "###############################################"
        echo "MODEL: ${MODEL}"
        echo "DATASET: ${DATASET}"
        echo "TARGET_CLASSES: ${TARGET_CLASSES[@]}"
        echo "TARGET_CLASS: ${TARGET_CLASS}"
        echo "###############################################"
        echo "###############################################"
        printf "\n\n\n\n\n"
        for SEED in 1
            do
                SAVE_DIR=results/logs/model_${MODEL}/dataset_${DATASET}/shots_${SHOTS}/target_${TARGET_CLASS}/seed_${SEED}
                if [ ! -d "$SAVE_DIR" ]; then
                    echo "Log folder="${SAVE_DIR}" does not exist."
                    exit 1
                fi     
                python main.py \
                --root ${DATASET_ROOT} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${SAVE_DIR} \
                --eval-only \
                --model-dir ${SAVE_DIR} \
                --load-epoch 50 \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} \
                BACKDOOR.TARGET_CLASS ${TARGET_CLASS} \
                BACKDOOR.POISON_PERCENTAGE 5 \
                BACKDOOR.NOISE_EPS 8 \
                MODEL_NAME ${MODEL} \
                MODEL_ROOT ${MODEL_ROOT} \
                DATASET_NAME ${DATASET} 
            done
    done
