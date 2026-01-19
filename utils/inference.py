import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def pl_bert_inference(model, test_dataLoader, device='cuda'):
    preds_list = []
    probs_list = []
    with torch.no_grad():
        for batch in test_dataLoader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            max_probs, _ = probs.max(dim=-1)
            preds_list.append(pred.cpu())
            probs_list.append(max_probs.cpu())
    preds_arr = torch.cat(preds_list).numpy()
    probs_arr = torch.cat(probs_list).numpy()
    return preds_arr, probs_arr


def learned_sketch_inference(model, x, label, dim):
    rows = np.arange(dim)
    scores = model[rows, x, :]
    sums = np.sum(scores, axis=0)
    sums_tensor = torch.from_numpy(sums)
    soft_max_sums = F.softmax(sums_tensor, dim=0)
    pred = int(np.argmax(soft_max_sums))
    pred_score = float(soft_max_sums[pred])
    return pred, pred_score