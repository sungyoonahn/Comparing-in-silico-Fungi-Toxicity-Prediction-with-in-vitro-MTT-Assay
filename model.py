import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model


from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassMatthewsCorrCoef, MulticlassAUROC
from transformers import BertModel, BertTokenizer
from utils import *


tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)


def load_dataset(dataset_dir):
    train_set = pd.read_csv(dataset_dir+"train.csv")
    valid_set = pd.read_csv(dataset_dir+"valid.csv")
    test_set = pd.read_csv(dataset_dir+"test.csv")

    return train_set, valid_set, test_set

class CustomProteinDataset(Dataset):
    def __init__(self, sequence, label, tokenizer, id):
        self.sequence = sequence
        self.label = label
        self.tokenizer = tokenizer
        self.id = id

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return {"sequence": self.sequence[idx], "labels": self.label[idx], "ids": self.id[idx]}

def collate_fn(data):
    embs = [d['sequence'] for d in data]
    tokens = tokenizer(embs, padding='longest', return_tensors='pt', max_length=1024, truncation=True) 
    ids = torch.tensor([d['ids'] for d in data])
    labels = torch.tensor([d['labels'] for d in data])
    # encodings = tokenizer(text = tjdgs["attention_mask"])
    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "ids": ids, "seqs": embs}

class CustomProteinDataset_inference(Dataset):
    def __init__(self, sequence, label, tokenizer, id):
        self.sequence = sequence
        self.label = label
        self.tokenizer = tokenizer
        self.id = id

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return {"sequence": self.sequence[idx], "labels": self.label[idx], "ids": self.id[idx]}

class CustomProtBERTModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomProtBERTModel, self).__init__()

        self.model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.final_proj = nn.Linear(self.model.config.hidden_size, self.num_labels)

    def forward(self,  input_ids=None, attention_mask=None, return_attentions=None):
        BERT_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=return_attentions)
        x = torch.mean(BERT_outputs.last_hidden_state, dim=1)
        final_hidden_embedding = self.dropout(x)
        logits = self.final_proj(final_hidden_embedding)
        softmax_logits = self.softmax(logits)
        return softmax_logits, final_hidden_embedding, BERT_outputs.attentions
   
def train(model, train_loader, optim, criterion, epoch, scheduler):
    model.train()

    losses = []
    count = 0
    LABELS = torch.tensor(()).to(device).int()
    PREDS = torch.tensor(()).to(device)
    IDS = torch.tensor(()).to(device)
    for batch in tqdm(train_loader, desc="Train"):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        ids = batch["ids"].to(device)
        logits, final_features, attentions = model(input_ids, attention_mask, None)

        # compute loss
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()
        losses.append(loss.item())

        count += 1
        LABELS = torch.cat((LABELS, labels))
        PREDS = torch.cat((PREDS, logits))
        IDS = torch.cat((IDS, ids))
    scheduler.step()
    print("lr: ", optim.param_groups[0]['lr'],"\n")
    acc, f1, mcc, auroc = eval_metrics(LABELS, PREDS, args.n_classes)

    avg_loss = np.mean(losses)
    avg_acc = acc.cpu().numpy()
    avg_f1 = f1.cpu().numpy()
    avg_mcc = mcc.cpu().numpy()
    avg_auroc = auroc.cpu().numpy()
    return [avg_loss, avg_acc, avg_f1, avg_mcc, avg_auroc]

def eval(model, valid_loader, criterion, epoch, type):
    model.eval()
    losses = []
    count = 0
    LABELS = torch.tensor(()).to(device).int()
    PREDS = torch.tensor(()).to(device)
    IDS = torch.tensor(()).to(device)

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Valid"):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch["ids"].to(device)
            logits, final_features, attentions = model(input_ids, attention_mask, None)

            if count == 0:
                total_final_features = final_features
                total_labels = labels

            else:
                total_final_features = torch.cat([total_final_features, final_features], dim=0)
                total_labels = torch.cat((total_labels, labels), dim=0)


            count += 1
            LABELS = torch.cat((LABELS, labels))
            PREDS = torch.cat((PREDS, logits))
            IDS = torch.cat((IDS, ids))
            # compute loss
            loss = criterion(logits, labels)
            # get metrics
            losses.append(loss.item())




    # plot output
    tsne_plot(total_final_features.cpu().numpy(), total_labels.cpu().numpy(), type, epoch)

    acc, f1, mcc, auroc = eval_metrics(LABELS, PREDS, args.n_classes)

    # average scores
    avg_loss = np.mean(losses)
    avg_acc = acc.cpu().numpy()
    avg_f1 = f1.cpu().numpy()
    avg_mcc = mcc.cpu().numpy()
    avg_auroc = auroc.cpu().numpy()

    return [avg_loss, avg_acc, avg_f1, avg_mcc, avg_auroc]

def infer(model, infer_loader, save_pth):
    model.eval()
    count = 0
    LABELS = torch.tensor(()).to(device).int()
    PREDS = torch.tensor(()).to(device)
    IDS = torch.tensor(()).to(device)

    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Inference"):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch["ids"].to(device)

            logits, final_features, attentions = model(input_ids, attention_mask, None)

            if count == 0:
                total_final_features = final_features
                total_labels = labels

            else:
                total_final_features = torch.cat([total_final_features, final_features], dim=0)
                total_labels = torch.cat((total_labels, labels), dim=0)

            count += 1
            # compute loss
            # loss = criterion(logits, labels)
            # get metrics

            LABELS = torch.cat((LABELS, labels))
            PREDS = torch.cat((PREDS,logits))
            IDS = torch.cat((IDS, ids))

    # plot output
    tsne_plot_inference(total_final_features.cpu().numpy(), total_labels.cpu().numpy(), save_pth)

    df = pd.DataFrame(IDS.cpu().numpy(), columns=["tensor id"]).astype('int64')
    df2 = pd.DataFrame(PREDS.cpu().numpy())
    df3 = pd.concat([df, df2], axis=1)
    print(df3)

    return df3


def infer_attention_map(model, infer_loader, save_pth):
    model.eval()
    count = 0
    LABELS = torch.tensor(()).to(device).int()
    PREDS = torch.tensor(()).to(device)
    IDS = torch.tensor(()).to(device)

    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Inference"):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch["ids"].to(device)
            seqs = batch['seqs']

            logits, final_features, attentions = model(input_ids, attention_mask, True)

            if count == 0:
                total_final_features = final_features
                total_labels = labels
                total_attentions = attentions
                total_seqs = seqs
                

            else:
                total_final_features = torch.cat([total_final_features, final_features], dim=0)
                total_labels = torch.cat((total_labels, labels), dim=0)
                total_attentions = np.append(total_attentions,attentions)
                total_seqs = np.concatenate((total_seqs, seqs), axis=0)

            count += 1


            LABELS = torch.cat((LABELS, labels))
            PREDS = torch.cat((PREDS,logits))
            IDS = torch.cat((IDS, ids))


    return total_attentions, total_seqs


def eval_metrics(y_true, y_pred, n_classes):
    acc_metric = MulticlassAccuracy(num_classes=n_classes).to(device)
    F1_metric = MulticlassF1Score(num_classes=n_classes).to(device)
    mcc_metric = MulticlassMatthewsCorrCoef(num_classes=n_classes).to(device)
    auroc_metric = MulticlassAUROC(num_classes=n_classes, thresholds=None).to(device)
    acc = acc_metric(y_pred, y_true)
    f1 = F1_metric(y_pred, y_true)
    mcc = mcc_metric(y_pred, y_true)
    auroc = auroc_metric(y_pred, y_true)
    return acc, f1, mcc, auroc


# Function for saving the model
def save_model(model, model_path):
    path = model_path
    torch.save(model.state_dict(), path)
    print("model saved!!!")
    print("model saved at: ", path)

def load_model(model, model_path):
    PATH = model_path
    model.load_state_dict(torch.load(PATH))
    return model 

def freeze_layers(model):
    
    modules = [model.embeddings, *model.encoder.layer[:11]]
    for module in modules: 
        for param in module.parameters():
            param.requires_grad = False
        
    return model


