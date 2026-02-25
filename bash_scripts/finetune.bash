
DATASETS=("shhs" "mros")
LABELS=("Stage" "Arousal" "Hypopnea" "Oxygen Desaturation")

TRAIN_PCTS=(1.0)  

declare -A MODELS

MODELS["dino_ours"]="last.ckpt|all"

for model_name in "${!MODELS[@]}"; do

    IFS='|' read -r ckpt_path use_backbone <<< "${MODELS[$model_name]}"
    
    for dataset in "${DATASETS[@]}"; do
        for label in "${LABELS[@]}"; do
            for pct in "${TRAIN_PCTS[@]}"; do
                echo "===== Model: ${model_name}, Dataset: ${dataset}, Label: ${label}, Pct: ${pct} ====="
                
                CUDA_VISIBLE_DEVICES=0,1,2,3 python main_finetune.py \
                    --train_data_pct ${pct} \
                    --max_steps 500 \
                    --use_which_backbone "${use_backbone}" \
                    --model_name "${model_name}" \
                    --ckpt_path "${ckpt_path}" \
                    --lr 0.1 \
                    --eval_label "${label}" \
                    --num_devices 4 \
                    --data_source both \
                    --include_datasets "${dataset}" \
                    --downstream_dataset_name "${dataset}"
            done
        done
    done
done
