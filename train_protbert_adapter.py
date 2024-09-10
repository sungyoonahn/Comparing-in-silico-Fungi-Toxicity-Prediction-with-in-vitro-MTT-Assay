import pandas as pd
import numpy as np
import copy
import os
import argparse
import random
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassMatthewsCorrCoef, MulticlassAUROC
import warnings

from transformers import BertModel, BertTokenizer
# from datasets import Dataset
# pd.set_option('display.max_columns', None)
#
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-d', default="datasets/HMP_1024_4class_test_new.csv", help='input csv file', type=str)
parser.add_argument('--out_csv_dir', '-o', default="test_output.csv", help='output csv file save name', type=str)
parser.add_argument('--n_classes', '-n', default=6, help='number of classes', type=int)
parser.add_argument('--freeze_layer', '-f', default=20, help='number of layers to freeze', type=int)
parser.add_argument('--batch_size', '-b', default=4, help='batch size', type=int)
parser.add_argument('--learning_rate', '-l', default=1e-4, help='learning rate', type=float)
parser.add_argument('--epoch', '-e', default=20, help='epoch', type=int)
parser.add_argument('--local_save_path', '-sv', default='/', help="local save path", type=str)
parser.add_argument('--fine_tune', '-ft', default=False, help='true or false for best weights', type=bool)
parser.add_argument('--mode', '-m', default="train", help="Choose mode, either train or instance")

args = parser.parse_args()
# ### hyper parameter
# df = pd.read_csv(args.data_dir)
# train_results_csv_save_path = args.out_csv_dir
# freeze_layers = args.freeze_layer
N_classes = args.n_classes
# BATCH_SIZE = args.batch_size
# learning_rate = args.learning_rate
# EPOCH = args.epoch
#
#
# print("input path: ", args.data_dir)
# print("output path: ", args.out_csv_dir)
# print("layers froze: ", args.freeze_layer)
# print("classes to train: ", args.n_classes)
# print("batch size: ", args.batch_size)
# print("learning rate: ", args.learning_rate)

# tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
# model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")

# Split dataset in to equal parts 7:2:1 for train:validation:test

warnings.filterwarnings("ignore")

main_train_csv_folder_path = "datasets/HMP_Swissprot_0522_batch0_3000/"
main_output_save_folder_path = "results/test/"
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


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
        # print(self.sequence)
        encodings = self.tokenizer(str(self.sequence[idx]), max_length=1024, padding="max_length", truncation=True)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings["attention_mask"])
        labels = torch.tensor(self.label[idx])
        ids = torch.tensor(self.id[idx])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "ids": ids}


class CustomProteinDataset_inference(Dataset):
    def __init__(self, sequence, label, tokenizer, id):
        self.sequence = sequence
        self.label = label
        self.tokenizer = tokenizer
        self.id = id

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        # print(self.sequence)
        encodings = self.tokenizer(str(self.sequence[idx]), max_length=1024, padding="max_length", truncation=True)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings["attention_mask"])
        labels = torch.tensor(self.label[idx])
        ids = torch.tensor(self.id[idx])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "ids": ids}

class CustomProtBERTModel(nn.Module):
    def __init__(self, model, num_labels):
        super(CustomProtBERTModel, self).__init__()

        self.model = model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.final_proj = nn.Linear(self.model.config.hidden_size, self.num_labels)

    def forward(self,  input_ids=None, attention_mask=None):
        BERT_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = torch.mean(BERT_outputs[0], dim=1)
        final_hidden_embedding = self.dropout(x)
        logits = self.final_proj(final_hidden_embedding)
        softmax_logits = self.softmax(logits)
        return softmax_logits, final_hidden_embedding

def train(model, train_loader, optim, criterion, epoch):
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
        logits, final_features = model(input_ids, attention_mask)
        # _, preds = torch.max(logits, 1)

        # compute loss
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()
        losses.append(loss.item())

        # if count == 0:
        #     total_final_features = final_features
        #     total_labels = labels
        #
        # else:
        #     total_final_features = torch.cat([total_final_features, final_features], dim=0)
        #     total_labels = torch.cat((total_labels, labels), dim=0)

        count += 1
        LABELS = torch.cat((LABELS, labels))
        PREDS = torch.cat((PREDS, logits))
        IDS = torch.cat((IDS, ids))


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
            logits, final_features = model(input_ids, attention_mask)

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

def infer(model, valid_loader):
    model.eval()
    # losses = []
    count = 0
    LABELS = torch.tensor(()).to(device).int()
    PREDS = torch.tensor(()).to(device)
    IDS = torch.tensor(()).to(device)

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Inference"):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch["ids"].to(device)

            logits, final_features = model(input_ids, attention_mask)

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


    acc, f1, mcc, auroc = eval_metrics(LABELS, PREDS, args.n_classes)
    print("ACCURACY\t",acc.cpu().numpy()*100)
    print("F1-SCORE\t",f1.cpu().numpy())
    print("MCC\t",mcc.cpu().numpy())
    print("AUROC\t",auroc.cpu().numpy())

    # print(IDS.cpu().numpy())
    # df['label'] = df['label'].astype('int64')
    # df = pd.DataFrame(IDS.cpu().numpy(), columns=["tensor id"]).astype('int64')
    # df2 = pd.DataFrame(PREDS.cpu().numpy(), columns=["normal", "secretion", "resistance", "toxin", "anti-toxin", "virulence", "invasion", "kill", "phage"])
    # df3 = pd.concat([df, df2], axis=1)
    # print(df3)
    # plot output
    tsne_plot_inference(total_final_features.cpu().numpy(), total_labels.cpu().numpy())
    # average scores
#    avg_acc = np.mean(ACC)
#    avg_f1 = np.mean(F1)
#    avg_mcc = np.mean(MCC)
#    avg_auroc = np.mean(AUROC)


    # return df3

def eval_metrics(y_true, y_pred, n_classes):
    acc_metric = MulticlassAccuracy(num_classes=n_classes).to(device)
    F1_metric = MulticlassF1Score(num_classes=n_classes).to(device)
    mcc_metric = MulticlassMatthewsCorrCoef(num_classes=n_classes).to(device)
    auroc_metric = MulticlassAUROC(num_classes=n_classes, thresholds=None).to(device)
    # y_true = torch.flatten(y_true)
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

def load_model(model, model_path):
    PATH = model_path
    model.load_state_dict(torch.load(PATH))
    return model

def tsne_plot(features, true_labels, type, epoch):
    torch.cuda.empty_cache()
    save_path = main_output_save_folder_path+"results_visualization/"+type+str(epoch)+".png"
    tsne_result = TSNE(n_components=2, perplexity=150).fit_transform(features)
    colormap = plt.cm.gist_ncar
    colorst = [colormap(i) for i in np.linspace(0, 0.9,len(true_labels))] 
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=true_labels)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE Visualization of Classification Results")
    plt.savefig(save_path)

def tsne_plot_inference(features, true_labels):
    torch.cuda.empty_cache()
    perplexities = [40,45,50,100, 150, 200]
    for i in perplexities:
        tsne_result = TSNE(n_components=2, perplexity=i).fit_transform(features)
        save_path = "results/240202_processed_data/inference_results/inference_perplexity_"+str(i)+".png"

        plt.figure(figsize=(10, 6))
        
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c = true_labels)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title("t-SNE Visualization of Classification Results perplexity = "+str(i))
        plt.savefig(save_path)

def freeze_layers(model):
    
    modules = [model.embeddings, *model.encoder.layer[:11]]
    for module in modules: 
        for param in module.parameters():
            param.requires_grad = False
        
    return model


if __name__ == "__main__":
    # initialize device - cuda if available cpu if else
    device = torch.device('cuda')
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # initialize model and tokenizer and device
    model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
    
    
    # # added lora
    config = LoraConfig(
        inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value", "key"]
    )
    model = get_peft_model(model, config)
    # model = freeze_layers(model)
    
    print_trainable_parameters(model)

    model = CustomProtBERTModel(model, args.n_classes)


    print_trainable_parameters(model)
    print(model)

    
    # for name, param in model.named_parameters():
    #     print(f'{name}: {param.requires_grad}')
    
    if args.fine_tune == True:
        print("Fine Tune Model")
        print("Loading Best Weights ... ")
        best_weights_path = "results/240202_processed_data/"
        model = load_model(model, best_weights_path+"18_loss_1.5476.pth")
        model.to(device)
        print(device)
    else:
        print("Training From Pretrained Weights")
        print_trainable_parameters(model)
        model.to(device)
        print(device)



    # If train, load all dataset
    # If inference load inference data
    if args.mode == "inference":
        print(device)
        print("Inference Mode")
        dataset_dir = main_train_csv_folder_path
        # save_dir = main_output_save_folder_path
        save_dir = "results/240202_processed_data/inference_results/"
        # infer_df = pd.read_csv(dataset_dir + "Salmonella_typhimurium_(strainLT2).csv")
        infer_df = pd.read_csv(dataset_dir + "test.csv")
        infer_df["sequence"] = infer_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
        infer_df['sequence'] = infer_df.apply(lambda row: " ".join(row["sequence"]), axis=1)
        infer_dataset = CustomProteinDataset(infer_df["sequence"], infer_df["label"], tokenizer, infer_df["tensor id"])
        infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=True)

        infer_results = infer(model, infer_loader)

        # print(infer_results)
        # infer_results.to_csv(save_dir + "tremble_test_results.csv")


    else:
        # read dataset
        dataset_dir = main_train_csv_folder_path
        train_df, valid_df, test_df = load_dataset(dataset_dir)

        # Preprocess inputs   
        # Replace uncommon AAs with "X"
        train_df["sequence"] = train_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
        valid_df["sequence"] = valid_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
        test_df["sequence"] = test_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)

        # Add spaces between each amino acid for PT5 to correctly use them
        train_df['sequence'] = train_df.apply(lambda row: " ".join(row["sequence"]), axis=1)
        valid_df['sequence'] = valid_df.apply(lambda row: " ".join(row["sequence"]), axis=1)
        test_df['sequence'] = test_df.apply(lambda row: " ".join(row["sequence"]), axis=1)

        # tokenize sequences
        train_dataset = CustomProteinDataset(train_df["sequence"], train_df["label"], tokenizer, train_df["tensor id"])
        valid_dataset = CustomProteinDataset(valid_df["sequence"], valid_df["label"], tokenizer,valid_df["tensor id"])
        test_dataset = CustomProteinDataset(test_df["sequence"], test_df["label"], tokenizer, test_df["tensor id"])

        # feed sequences to Dataloader for trainable input
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        # results df for reading outputs
        results_df = pd.DataFrame(columns=['epoch', 'loss', 'ACC', 'F1', 'MCC', 'AUROC'])


        min_valid_loss = 10
        optim = AdamW(model.parameters(), lr=args.learning_rate)
        criterion = CrossEntropyLoss()

        for epoch in tqdm(range(args.epoch), desc="Epochs", unit="epoch"):

            train_results = train(model, train_loader, optim, criterion, epoch)
            valid_results = eval(model, valid_loader, criterion, epoch, "valid")

            if min_valid_loss > valid_results[0]:
                test_results = eval(model, test_loader, criterion, epoch, "test")
                test_results.insert(0, "test")
                min_valid_loss = valid_results[0]
                save_path = main_output_save_folder_path+str(epoch)+"_loss_{:.4f}.pth".format(float(valid_results[0]))
                save_model(model, model_path=save_path)
                save_model(model, model_path=main_output_save_folder_path+"best_weights.pth")

            train_results.insert(0, "train")
            valid_results.insert(0, "valid")

            results_list = [[epoch], train_results, valid_results, test_results]
            # print(results_list)
            results_df = pd.concat([results_df, pd.DataFrame(results_list, columns=['epoch', 'loss', 'ACC', 'F1', 'MCC', 'AUROC'])])
            results_df.to_csv(main_output_save_folder_path+"results.csv", index=False)
