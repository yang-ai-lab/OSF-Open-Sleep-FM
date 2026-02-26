from config import *


# Uni-encoder models (simclr, dino, mae, vqvae, ar, etc.)
TRAIN_EDF_COLS_UNI_ENC = [ECG, EMG_Chin, EMG_LLeg, EMG_RLeg,
                  ABD, THX, NP, SN, 
                  EOG_E1_A2, EOG_E2_A1,EEG_C3_A2, EEG_C4_A1,
            ]
TRAIN_EDF_COLS_MULTI_ENC = [ECG, 
                  ABD, THX, NP, SN, 
                  EMG_Chin, EMG_LLeg, EMG_RLeg,
                  EOG_E1_A2, EOG_E2_A1,EEG_C3_A2, EEG_C4_A1,
            ]
TRAIN_EDF_COLS_TYPE3 = [ECG, ABD, THX, NP, SN]
TRAIN_EDF_COLS_TYPE4 = [ECG, ABD, THX]


MONITOR_TYPE_MAP = {
    "main": TRAIN_EDF_COLS_UNI_ENC,
    "type3": TRAIN_EDF_COLS_TYPE3,
    "type4": TRAIN_EDF_COLS_TYPE4,
}
STAGE2_LABEL_PATH_WITH_PATHHEAD = "/Yourpath/merged_patient_csv_splits_real"  
CKPT_PATH = "/Yourpath/unimodal_zitao_pretrain"
MODEL_LIST = ["dino_ours"]

AUGMENTATION_MAP = {
    "dino_ours": "chan_then_pcspan",
}
SPLIT_DATA_FOLDER = "/Yourpath/sleep_postprocessed_data"
PRETRAIN_VAL_DATASET_LIST = ['shhs']
NEED_NORM_COL = [HR, SPO2, OX]
