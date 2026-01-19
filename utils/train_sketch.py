import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import utils.preprocessing as preprocessing_utils
import utils.datasets as datasets_utils
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments, Trainer

def train(model, frequency_filter, loader, learning_rate=0.001, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        model.conv1.weight.copy_(frequency_filter)
        model.conv1.bias.data.zero_()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.cuda()
            yb = yb.cuda()

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            epoch_loss += loss.item() * batch_size

            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += batch_size

        avg_loss = epoch_loss / total
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")
    return model


def train_frequency_filter(df, max_length, label_size):
    idx_cnt = dict()
    idx_label_cnt = dict()
    
    for i in range(max_length):
    	idx_cnt[i] = dict()
    	idx_label_cnt[i] = dict()
    	for j in range(label_size):
    		idx_label_cnt[i][j] = dict()
    
    idx_label_ratio = dict()
    label_cnt = dict()
    
    for _, row in df.iterrows():
    	data = preprocessing_utils.sketch_preprocessing(row['x'], max_length)
    	label = row['y']
    
    	if label not in label_cnt:
    		label_cnt[label] = 0
    	label_cnt[label] += 1
    	
    	for idx, d in enumerate(data[: max_length]):
    		val = data[idx]
    	
    		if val not in idx_cnt[idx]:
    			idx_cnt[idx][val] = 0
    		idx_cnt[idx][val] += 1
    
    		if val not in idx_label_cnt[idx][label]:
    			idx_label_cnt[idx][label][val] = 0
    		idx_label_cnt[idx][label][val] += 1

    
    for idx in range(max_length):
    	idx_label_ratio[idx] = dict()
    	for l in range(label_size):
    		idx_label_ratio[idx][l] = dict()
    		for k in idx_label_cnt[idx][l].keys():
    			idx_label_ratio[idx][l][k] = idx_label_cnt[idx][l][k] / idx_cnt[idx][k]

    
    replace_conv = [[[0.0 for k in range(3001)]for j in range(max_length)]for i in range(label_size)]
    replace_conv = np.array(replace_conv)
    
    for label in range(label_size):
        for idx in range(max_length):
            for k, value in idx_label_ratio[idx][label].items():
                if 0 <= val < 3001:
                    replace_conv[label][idx][k] = value
    
    replace_conv_torch = torch.tensor(replace_conv, dtype=torch.float32)
    replace_conv_torch = replace_conv_torch.unsqueeze(1)
    return replace_conv, replace_conv_torch