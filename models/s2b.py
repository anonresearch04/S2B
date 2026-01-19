import os
import torch
import dotenv
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import utils.train_sketch as train_sketch_utils
import utils.train_pl_bert as train_pl_bert_utils
import models.sketch_model as sketch_model
from transformers import BertForSequenceClassification, BertTokenizer
import utils.datasets as datasets_utils
import models.pl_bert_model as pl_bert_model
import utils.inference as inference_utils
import utils.threshold as threshold_utils
import utils.preprocessing as preprocessing_utils

from sklearn.metrics import f1_score

class S2B():
    def __init__(self, args, mode="s2b-c", openworld=False):
        self.args = args
        self.mode = mode
        self.openworld = openworld
        self.pl_bert_tokenizer = BertTokenizer.from_pretrained(self.args["BASE_BERT_MODEL"], cache_dir="./hf_cache", local_files_only=True)
        self.frequency_filter = None
        self.learned_filter = None
        self.pl_bert = None
        self.sketch_finalize_threshold = {}
        self.sketch_ood_threshold = {}
        self.gamma_0 = 1.0
        self.gamma_1 = 0.75
        self.theta_2 = 0.97
        self.now = int(datetime.now().timestamp())
        if(args["IS_TRAIN"]):
            self.output_path = os.path.join(self.args["OUTPUT_PATH"], str(self.now))
            os.makedirs(self.output_path, exist_ok=True)


    def forward(self):
        if(not self.openworld):
            test_df = pd.read_parquet(self.args["TEST_DF"])
            if self.mode == "s2b-c":
                return self.forward_s2b_c_closeworld(test_df)
            elif self.mode == "s2b-s":
                return self.forward_s2b_s_closeworld(test_df)
            elif self.mode == "s2b-b":
                return self.forward_s2b_b_closeworld(test_df)
        else:
            test_df = pd.read_parquet(self.args["OPEN_TEST_DF"])
            if self.mode == "s2b-c":
                return self.forward_s2b_c_openworld(test_df)
            elif self.mode == "s2b-s":
                return self.forward_s2b_s_openworld(test_df)
            elif self.mode == "s2b-b":
                return self.forward_s2b_b_openworld(test_df)
        return x


    def load_model(self):
        model = sketch_model.SketchModel(self.args["DIM"], self.args["LABEL"])
        model.load_state_dict(torch.load(self.args["SKETCH_MODEL_PATH"]))
        self.learned_filter = model.get_sketch()
        self.pl_bert = BertForSequenceClassification.from_pretrained(self.args["BERT_MODEL_PATH"], output_attentions=True)
        device = torch.device(self.args["DEVICE"])
        self.pl_bert.to(device)


    def forward_s2b_c_openworld(self, df):
        preds = []
        reals = []
        for i in tqdm(range(len(df))):
            item = df.iloc[[i]]
            x = preprocessing_utils.sketch_preprocessing(item['x'].values[0], self.args["DIM"])
            y = item['y'].values[0]
            sketch_pred, sketch_proba = inference_utils.learned_sketch_inference(self.learned_filter, x, self.args["LABEL"], self.args["DIM"])
            if(sketch_proba < self.sketch_ood_threshold[sketch_pred]):
                pred = -1
            elif(sketch_proba >= self.sketch_finalize_threshold[sketch_pred] * self.gamma_1):
                pred = sketch_pred
            else:
                dataset = datasets_utils.get_pl_bert_dataset(self.pl_bert_tokenizer, item, self.args["DIM"])
                test_loader = datasets_utils.get_pl_bert_dataLoader(dataset, self.args["PL_BERT_BATCH_SIZE"], shuffle=False)
                pl_bert_pred, pl_bert_proba = inference_utils.pl_bert_inference(self.pl_bert, test_loader)
                if(pl_bert_proba[0] < self.theta_2):
                    pred = -1
                else:
                    pred = pl_bert_pred[0]
            preds.append(pred)
            reals.append(y)
        print(f1_score(reals, preds, average='macro'))
        return preds, reals


    def forward_s2b_s_openworld(self, df):
        preds = []
        reals = []
        for i in tqdm(range(len(df))):
            item = df.iloc[i]
            x = preprocessing_utils.sketch_preprocessing(item['x'], self.args["DIM"])
            y = item['y']
            pred, pred_score = inference_utils.learned_sketch_inference(self.learned_filter, x, self.args["LABEL"], self.args["DIM"])
            if(pred_score >= self.sketch_ood_threshold[pred]):
                pred = pred
            else:
                pred = -1
            preds.append(pred)
            reals.append(y)
        print(f1_score(reals, preds, average='macro'))
        return preds, reals


    def forward_s2b_b_openworld(self, df):
        preds = []
        reals = []
        self.pl_bert.eval()
        for i in tqdm(range(len(df))):
            item = df.iloc[[i]]
            pl_bert_dataset = datasets_utils.get_pl_bert_dataset(self.pl_bert_tokenizer, item, self.args["DIM"])
            test_loader = datasets_utils.get_pl_bert_dataLoader(pl_bert_dataset, self.args["PL_BERT_BATCH_SIZE"], shuffle=False)
            pred, pred_proba = inference_utils.pl_bert_inference(self.pl_bert, test_loader)
            pred = pred.copy()
            pred[pred_proba < self.theta_2] = -1
            preds.extend(pred)
            reals.extend(item['y'])
        print(f1_score(reals, preds, average='macro'))
        return preds, reals


    def forward_s2b_c_closeworld(self, df):
        preds = []
        reals = []
        for i in tqdm(range(len(df))):
            item = df.iloc[[i]]
            x = preprocessing_utils.sketch_preprocessing(item['x'].values[0], self.args["DIM"])
            y = item['y'].values[0]
            sketch_pred, sketch_proba = inference_utils.learned_sketch_inference(self.learned_filter, x, self.args["LABEL"], self.args["DIM"])
            if(sketch_proba >= self.sketch_finalize_threshold[sketch_pred] * self.gamma_1):
                pred = sketch_pred
            else:
                dataset = datasets_utils.get_pl_bert_dataset(self.pl_bert_tokenizer, item, self.args["DIM"])
                test_loader = datasets_utils.get_pl_bert_dataLoader(dataset, self.args["PL_BERT_BATCH_SIZE"], shuffle=False)
                pl_bert_pred, pl_bert_proba = inference_utils.pl_bert_inference(self.pl_bert, test_loader)
                pred = pl_bert_pred[0]
            preds.append(pred)
            reals.append(y)
        print(f1_score(reals, preds, average='macro'))
        return preds, reals


    def forward_s2b_s_closeworld(self, df):
        preds = []
        reals = []
        for i in tqdm(range(len(df))):
            item = df.iloc[i]
            x = preprocessing_utils.sketch_preprocessing(item['x'], self.args["DIM"])
            y = item['y']
            pred, _ = inference_utils.learned_sketch_inference(self.learned_filter, x, self.args["LABEL"], self.args["DIM"])
            preds.append(pred)
            reals.append(y)
        print(f1_score(reals, preds, average='macro'))
        return preds, reals
        

    def forward_s2b_b_closeworld(self, df):
        preds = []
        reals = []
        self.pl_bert.eval()
        for i in tqdm(range(len(df))):
            item = df.iloc[[i]]
            pl_bert_dataset = datasets_utils.get_pl_bert_dataset(self.pl_bert_tokenizer, item, self.args["DIM"])
            test_loader = datasets_utils.get_pl_bert_dataLoader(pl_bert_dataset, self.args["PL_BERT_BATCH_SIZE"], shuffle=False)
            pred, _ = inference_utils.pl_bert_inference(self.pl_bert, test_loader)
            preds.extend(pred)
            reals.extend(item['y'])
        print(f1_score(reals, preds, average='macro'))
        return preds, reals


    def setting_sketch_finalize_threshold(self):
        correct = {i: [] for i in range(self.args["LABEL"])}
        incorrect = {i: [] for i in range(self.args["LABEL"])}
        valid_df = pd.read_parquet(self.args["VALID_DF"])
        for i in tqdm(range(len(valid_df))):
            item = valid_df.iloc[i]
            x = preprocessing_utils.sketch_preprocessing(item['x'], self.args["DIM"])
            y = item['y']
            pred, pred_score = inference_utils.learned_sketch_inference(self.learned_filter, x, self.args["LABEL"], self.args["DIM"])
            if(pred == y):
                correct[pred].append(pred_score)
            else:
                incorrect[pred].append(pred_score)
        
        for i in range(self.args["LABEL"]):
            self.sketch_finalize_threshold[i] = threshold_utils.get_sketch_finalize_threshold(correct[i], incorrect[i])

    
    def setting_sketch_ood_threshold(self):
        correct = {i: [] for i in range(self.args["LABEL"])}
        incorrect = {i: [] for i in range(self.args["LABEL"])}
        valid_df = pd.read_parquet(self.args["VALID_DF"])
        for i in tqdm(range(len(valid_df))):
            item = valid_df.iloc[i]
            x = preprocessing_utils.sketch_preprocessing(item['x'], self.args["DIM"])
            y = item['y']
            pred, pred_score = inference_utils.learned_sketch_inference(self.learned_filter, x, self.args["LABEL"], self.args["DIM"])
            if(pred == y):
                correct[pred].append(pred_score)
            else:
                incorrect[pred].append(pred_score)
        for i in range(self.args["LABEL"]):
            self.sketch_ood_threshold[i] = threshold_utils.get_sketch_ood_threshold(correct[i], incorrect[i], sigma=self.gamma_0)


    def train_frequency_filter(self):
        train_df = pd.read_parquet(self.args["TRAIN_DF"])
        _, self.frequency_filter = train_sketch_utils.train_frequency_filter(train_df, self.args["DIM"], self.args["LABEL"])


    def train_learned_filter(self):
        train_df = pd.read_parquet(self.args["TRAIN_DF"])
        valid_df = pd.read_parquet(self.args["VALID_DF"])
        model = sketch_model.SketchModel(self.args["DIM"], self.args["LABEL"])
        dataset, loader = datasets_utils.get_learned_filter_dataLoader(train_df, self.args["DIM"], self.args["SKETCH_BATCH_SIZE"], shuffle=True)
        learned_model = train_sketch_utils.train(model, self.frequency_filter, loader, learning_rate=self.args["SKETCH_LR"], epochs=self.args["SKETCH_EPOCH"])
        torch.save(model.state_dict(), os.path.join(self.output_path, "learned_sketch_model.pth"))
        self.learned_filter = learned_model.get_sketch()


    def train_pl_bert(self):
        train_df = pd.read_parquet(self.args["TRAIN_DF"])
        valid_df = pd.read_parquet(self.args["VALID_DF"])
        model = pl_bert_model.get_model(self.args["BASE_BERT_MODEL"], self.args["LABEL"])
        train_dataset = datasets_utils.get_pl_bert_dataset(self.pl_bert_tokenizer, train_df, self.args["DIM"])
        validation_dataset = datasets_utils.get_pl_bert_dataset(self.pl_bert_tokenizer, valid_df, self.args["DIM"])
        train_dataLoader = datasets_utils.get_pl_bert_dataLoader(train_dataset, self.args["PL_BERT_BATCH_SIZE"], shuffle=True)
        validation_dataLoader = datasets_utils.get_pl_bert_dataLoader(validation_dataset, self.args["PL_BERT_BATCH_SIZE"], shuffle=False)
        self.pl_bert = train_pl_bert_utils.train(model, train_dataset, validation_dataset, self.output_path, self.args["PL_BERT_BATCH_SIZE"], self.args["PL_BERT_EPOCH"])
