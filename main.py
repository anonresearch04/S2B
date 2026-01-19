import os
import dotenv
import models.s2b as s2b_model
import utils.preprocessing as preprocessing_utils


dotenv.load_dotenv()
args = {
    "DIM": int(os.getenv("DIM")),
    "LABEL": int(os.getenv("LABEL")),
    "OPENWORLD": os.getenv("OPENWORLD", "False").lower() == "true",
    "IS_TRAIN" : os.getenv("IS_TRAIN", "False").lower() == "true",
    "MODE": os.getenv("MODE"),
    "TRAIN_DF": os.getenv("TRAIN_DF"),
    "VALID_DF": os.getenv("VALID_DF"),
    "TEST_DF": os.getenv("TEST_DF"),
    "OPEN_TEST_DF": os.getenv("OPEN_TEST_DF"),
    "RAW_TRAIN_DF": os.getenv("RAW_TRAIN_DF"),
    "RAW_VALID_DF": os.getenv("RAW_VALID_DF"),
    "RAW_TEST_DF": os.getenv("RAW_TEST_DF"),
    "OPEN_RAW_TEST_DF": os.getenv("OPEN_RAW_TEST_DF"),
    "RAW_X_LABEL": os.getenv("RAW_X_LABEL"),
    "RAW_Y_LABEL": os.getenv("RAW_Y_LABEL"),
    "DATASET_PATH": os.getenv("DATASET_PATH"),
    "BASE_BERT_MODEL": os.getenv("BASE_BERT_MODEL"),
    "PL_BERT_BATCH_SIZE": int(os.getenv("PL_BERT_BATCH_SIZE")),
    "PL_BERT_EPOCH": int(os.getenv("PL_BERT_EPOCH")),
    "PL_BERT_LR": float(os.getenv("PL_BERT_LR")),
    "SKETCH_BATCH_SIZE": int(os.getenv("SKETCH_BATCH_SIZE")),
    "SKETCH_LR" : float(os.getenv("SKETCH_LR")),
    "SKETCH_EPOCH": int(os.getenv("SKETCH_EPOCH")),
    "OUTPUT_PATH": os.getenv("OUTPUT_PATH"),
    "SKETCH_MODEL_PATH": os.getenv("SKETCH_MODEL_PATH"),
    "BERT_MODEL_PATH": os.getenv("BERT_MODEL_PATH"),
    "DEVICE": os.getenv("DEVICE"),
}

def preprocess():
    preprocessed_train_df = preprocessing_utils.get_preprocessed_df(args["RAW_TRAIN_DF"], args["RAW_X_LABEL"], args["RAW_Y_LABEL"], args["DIM"], args["DATASET_PATH"])
    preprocessed_valid_df = preprocessing_utils.get_preprocessed_df(args["RAW_VALID_DF"], args["RAW_X_LABEL"], args["RAW_Y_LABEL"], args["DIM"], args["DATASET_PATH"])
    preprocessed_test_df = preprocessing_utils.get_preprocessed_df(args["RAW_TEST_DF"], args["RAW_X_LABEL"], args["RAW_Y_LABEL"], args["DIM"], args["DATASET_PATH"])
    preprocessed_openworld_test_df = preprocessing_utils.get_preprocessed_df(args["OPEN_RAW_TEST_DF"], args["RAW_X_LABEL"], args["RAW_Y_LABEL"], args["DIM"], args["DATASET_PATH"])

def main():
    if(args["IS_TRAIN"]):
        s2b = s2b_model.S2B(args, mode=args["MODE"], openworld=args["OPENWORLD"])
        s2b.train_frequency_filter()
        s2b.train_learned_filter()
        s2b.train_pl_bert()
    else:
        s2b = s2b_model.S2B(args, mode=args["MODE"], openworld=args["OPENWORLD"])
        s2b.load_model()
        s2b.setting_sketch_ood_threshold()
        s2b.setting_sketch_finalize_threshold()
        s2b.forward()


if __name__ == "__main__":
    main()