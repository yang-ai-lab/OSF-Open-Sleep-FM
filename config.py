"""
Configuration constants for sleep data processing.
Contains dataset names, paths, channel definitions, and event labels.
"""
import pandas as pd
import numpy as np

# =============================================================================
# Dataset name constants
# =============================================================================
SHHS = 'shhs'
CHAT = 'chat'
MROS = 'mros'
CCSHS = 'ccshs'
CFS = 'cfs'
MESA = 'mesa'
SOF = 'sof'
WSC = 'wsc'
HSP = 'hsp'
NCHSDB = 'nchsdb'
STAGES = 'stages'
PATS = 'pats'
SHHS2 = 'shhs2'
NUMOM2B = 'numom2b'

# =============================================================================
# Data paths
# =============================================================================
META_PATH = '/scratch/besp/shared_data'

MASTER_SHHS = [META_PATH + "/" + SHHS + "/datasets/shhs-harmonized-dataset-0.21.0.csv"]
MASTER_CHAT = [META_PATH + "/" + CHAT + "/datasets/chat-harmonized-dataset-0.14.0.csv"]
MASTER_MROS = [META_PATH + "/" + MROS + "/datasets/mros-visit1-harmonized-0.6.0.csv"]
MASTER_CCSHS = [META_PATH + "/" + CCSHS + "/datasets/ccshs-trec-harmonized-0.8.0.csv"]
MASTER_CFS = [META_PATH + "/" + CFS + "/datasets/cfs-visit5-harmonized-dataset-0.7.0.csv"]
MASTER_MESA = [META_PATH + "/" + MESA + "/datasets/mesa-sleep-harmonized-dataset-0.7.0.csv"]
MASTER_SOF = [META_PATH + "/" + SOF + "/datasets/sof-visit-8-harmonized-dataset-0.8.0.csv"]
MASTER_WSC = [META_PATH + "/" + WSC + "/datasets/wsc-harmonized-dataset-0.7.0.csv"]
MASTER_HSP = [
    META_PATH + "/" + HSP + "/psg-metadata/I0001_psg_metadata_2025-05-06.csv",
    META_PATH + "/" + HSP + "/psg-metadata/I0002_psg_metadata_2025-05-06.csv",
    META_PATH + "/" + HSP + "/psg-metadata/I0003_psg_metadata_2025-05-06.csv",
    META_PATH + "/" + HSP + "/psg-metadata/I0004_psg_metadata_2025-05-06.csv",
    META_PATH + "/" + HSP + "/psg-metadata/I0006_psg_metadata_2025-05-06.csv",
]
MASTER_STAGES = [META_PATH + "/" + STAGES + "/metadata/stages-harmonized-dataset-0.3.0.csv"]
MASTER_NCHSDB = [META_PATH + "/" + NCHSDB + "/datasets/nchsdb-dataset-harmonized-0.3.0.csv"]
MASTER_PATS = [META_PATH + "/" + PATS + "/datasets/pats-harmonized-dataset-0.1.0.csv"]

MASTER_CSV_LIST = {
    'shhs': MASTER_SHHS,
    'chat': MASTER_CHAT,
    'mros': MASTER_MROS,
    'ccshs': MASTER_CCSHS,
    'cfs': MASTER_CFS,
    'mesa': MASTER_MESA,
    'sof': MASTER_SOF,
    'wsc': MASTER_WSC,
    'hsp': MASTER_HSP,
    'stages': MASTER_STAGES,
    'pats': MASTER_PATS,
    'nchsdb': MASTER_NCHSDB,
}

# =============================================================================
# Channel name constants
# =============================================================================
# ECG channels
ECG = 'ECG'
ECG1 = 'ECG1'
ECG2 = 'ECG2'
ECG3 = 'ECG3'
HR = 'HR'
PPG = 'PPG'

# Respiratory channels
SPO2 = 'SPO2'
OX = 'OX'
ABD = 'ABD'
THX = 'THX'
AF = 'AF'
NP = 'NP'
SN = 'SN'

# EOG channels
EOG_L = 'EOG_L'
EOG_R = 'EOG_R'
EOG_E1_A2 = 'EOG_E1_A2'
EOG_E2_A1 = 'EOG_E2_A1'

# EMG Leg channels
EMG_LLeg = 'EMG_LLeg'
EMG_RLeg = 'EMG_RLeg'
EMG_LLeg1 = 'EMG_LLeg1'
EMG_LLeg2 = 'EMG_LLeg2'
EMG_RLeg1 = 'EMG_RLeg1'
EMG_RLeg2 = 'EMG_RLeg2'
EMG_Leg = 'EMG_Leg'

# Sensor Leg channels
SENSOR_Leg = 'SENSOR_Leg'
SENSOR_LLeg = 'SENSOR_LLeg'
SENSOR_LLeg1 = 'SENSOR_LLeg1'
SENSOR_LLeg2 = 'SENSOR_LLeg2'
SENSOR_RLeg = 'SENSOR_RLeg'
SENSOR_RLeg1 = 'SENSOR_RLeg1'
SENSOR_RLeg2 = 'SENSOR_RLeg2'

# EMG Chin channels
EMG_Chin = 'EMG_Chin'
EMG_RChin = 'EMG_RChin'
EMG_LChin = 'EMG_LChin'
EMG_CChin = 'EMG_CChin'

# EEG channels (unipolar)
EEG_C3 = 'EEG_C3'
EEG_C4 = 'EEG_C4'
EEG_A1 = 'EEG_A1'
EEG_A2 = 'EEG_A2'
EEG_O1 = 'EEG_O1'
EEG_O2 = 'EEG_O2'
EEG_F3 = 'EEG_F3'
EEG_F4 = 'EEG_F4'

# EEG channels (bipolar/referenced)
EEG_C3_A2 = 'EEG_C3_A2'
EEG_C4_A1 = 'EEG_C4_A1'
EEG_F3_A2 = 'EEG_F3_A2'
EEG_F4_A1 = 'EEG_F4_A1'
EEG_O1_A2 = 'EEG_O1_A2'
EEG_O2_A1 = 'EEG_O2_A1'

# Other channels
FPZ = 'FPZ'
GROUND = 'GROUND'
POS = 'POS'

# =============================================================================
# Sampling frequencies (Hz)
# =============================================================================
FREQ_ECG = 128
FREQ_ECG1 = 128
FREQ_ECG2 = 128
FREQ_ECG3 = 128
FREQ_HR = 1
FREQ_PPG = 128

FREQ_SPO2 = 1
FREQ_OX = 1
FREQ_ABD = 8
FREQ_THX = 8
FREQ_AF = 8
FREQ_NP = 8
FREQ_SN = 32

FREQ_EOG_L = 64
FREQ_EOG_R = 64
FREQ_EOG_E1_A2 = 64
FREQ_EOG_E2_A1 = 64

FREQ_EMG_Leg = 64
FREQ_EMG_LLeg = 64
FREQ_EMG_RLeg = 64
FREQ_EMG_LLeg1 = 64
FREQ_EMG_LLeg2 = 64
FREQ_EMG_RLeg1 = 64
FREQ_EMG_RLeg2 = 64

FREQ_SENSOR_Leg = 64
FREQ_SENSOR_LLeg = 64
FREQ_SENSOR_LLeg1 = 64
FREQ_SENSOR_LLeg2 = 64
FREQ_SENSOR_RLeg = 64
FREQ_SENSOR_RLeg1 = 64
FREQ_SENSOR_RLeg2 = 64

FREQ_EMG_Chin = 64
FREQ_EMG_LChin = 64
FREQ_EMG_RChin = 64
FREQ_EMG_CChin = 64

FREQ_EEG_C3 = 64
FREQ_EEG_C4 = 64
FREQ_EEG_A1 = 64
FREQ_EEG_A2 = 64
FREQ_EEG_O1 = 64
FREQ_EEG_O2 = 64
FREQ_EEG_F3 = 64
FREQ_EEG_F4 = 64

FREQ_EEG_C3_A2 = 64
FREQ_EEG_C4_A1 = 64
FREQ_EEG_F3_A2 = 64
FREQ_EEG_F4_A1 = 64
FREQ_EEG_O1_A2 = 64
FREQ_EEG_O2_A1 = 64

FREQ_POS = 1

# =============================================================================
# Event annotation column names
# =============================================================================
EVENT_NAME_COLUMN = 'EVENT'
START_TIME_COLUMN = 'START_SEC'
END_TIME_COLUMN = 'END_SEC'

# =============================================================================
# Respiratory event names
# =============================================================================
RESPIRATORY_EVENT_CENTRAL_APNEA = 'Central Apnea'
RESPIRATORY_EVENT_OBSTRUCTIVE_APNEA = 'Obstructive Apnea'
RESPIRATORY_EVENT_MIXED_APNEA = 'Mixed Apnea'
RESPIRATORY_EVENT_HYPOPNEA = 'Hypopnea'
RESPIRATORY_EVENT_DESATURATION = 'Oxygen Desaturation'

# =============================================================================
# Limb movement event names
# =============================================================================
LIMB_MOVEMENT_ISOLATED = 'Limb Movement Isolated'
LIMB_MOVEMENT_PERIODIC = 'Limb Movement Periodic'
LIMB_MOVEMENT_ISOLATED_LEFT = 'Left Limb Movement Isolated'
LIMB_MOVEMENT_ISOLATED_RIGHT = 'Right Limb Movement Isolated'
LIMB_MOVEMENT_PERIODIC_LEFT = 'Left Limb Movement Periodic'
LIMB_MOVEMENT_PERIODIC_RIGHT = 'Right Limb Movement Periodic'

# =============================================================================
# Arousal event names
# =============================================================================
AROUSAL_EVENT_CLASSIC = 'Arousal'
AROUSAL_EVENT_RESPIRATORY = 'RERA'
AROUSAL_EVENT_EMG = 'EMG-Related Arousal'
