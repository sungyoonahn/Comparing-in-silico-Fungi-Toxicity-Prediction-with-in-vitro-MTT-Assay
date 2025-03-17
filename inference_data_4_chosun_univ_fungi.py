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
import datetime
from transformers import BertModel, BertTokenizer
# from datasets import Dataset
# pd.set_option('display.max_columns', None)
#
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-d', default="datasets/HMP_1024_4class_test_new.csv", help='input csv file', type=str)
parser.add_argument('--out_csv_dir', '-o', default="test_output.csv", help='output csv file save name', type=str)
parser.add_argument('--n_classes', '-n', default=5, help='number of classes', type=int)
parser.add_argument('--freeze_layer', '-f', default=20, help='number of layers to freeze', type=int)
parser.add_argument('--batch_size', '-b', default=1, help='batch size', type=int)
parser.add_argument('--learning_rate', '-l', default=1e-4, help='learning rate', type=float)
parser.add_argument('--epoch', '-e', default=20, help='epoch', type=int)
parser.add_argument('--local_save_path', '-sv', default='/', help="local save path", type=str)
parser.add_argument('--fine_tune', '-ft', default=True, help='true or false for best weights', type=bool)
parser.add_argument('--mode', '-m', default="inference", help="Choose mode, either train or inference")

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


# tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
# model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")

# Split dataset in to equal parts 7:2:1 for train:validation:test

warnings.filterwarnings("ignore")

### CHANGE ROOT FOLDER HERE ###
main_train_csv_folder_path = "sungyoon/env_project/datasets/fungi_inference/NCBI_fungi_csv_0218/"
main_output_save_folder_path = "sungyoon/env_project/results/fungi_inference/NCBI_fungi_csv_0218/"

def merge_inferenece(df, df_compare):
### BELOW CODE IS FOR MERGING INFERENCE RESULTS WITH THEIR INFORMATION

    print(df_compare)
    # df_compare = df_compare.drop('Unnamed: 0', axis=1)

    df_new = tensor_id_merge(df, df_compare)
    # print(df)

    # print(df_new)
    return df_new
    
def tensor_id_merge(df1, df2):
    df3 = pd.merge(df1, df2, on='tensor id', how='inner')

    return df3

def check_labels_count(df):
    print("checking count for each labels...\n-------")
    print(pd.Series(df["label"]).value_counts().reset_index())

def preds_check_labels_count(df):
    print("checking count for each labels...\n-------")
    print(pd.Series(df["max label"]).value_counts().reset_index())
    
def matching_labels(df):
    print("checking matching prediction and base results...'n")
    matching_labels = df["max label"] == df["label"]
    print(matching_labels)
    df["matching label"] = matching_labels
    
    return df
        
    
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
    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "ids": ids}



class CustomProtBERTModel(nn.Module):
    def __init__(self, model, num_labels):
        super(CustomProtBERTModel, self).__init__()

        self.model = model
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
   


def infer(model, valid_loader, save_pth):
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

            logits, final_features, attentions = model(input_ids, attention_mask, None)

            # print(len(attentions))
            if count == 0:
                total_final_features = final_features
                total_labels = labels

            else:
                total_final_features = torch.cat([total_final_features, final_features], dim=0)
                total_labels = torch.cat((total_labels, labels), dim=0)

            count += 1

            LABELS = torch.cat((LABELS, labels))
            PREDS = torch.cat((PREDS,logits))
            IDS = torch.cat((IDS, ids))



    # tsne_plot_inference(total_final_features.cpu().numpy(), total_labels.cpu().numpy(), save_pth)

    df = pd.DataFrame(IDS.cpu().numpy(), columns=["tensor id"]).astype('int64')
    df2 = pd.DataFrame(PREDS.cpu().numpy())
    df3 = pd.concat([df, df2], axis=1)
    print(df3)

    return df3





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

def tsne_plot_inference(features, true_labels, save_pth):
    torch.cuda.empty_cache()
    perplexities = [40,45,50,100, 150, 200]
    for i in perplexities:
        tsne_result = TSNE(n_components=2, perplexity=i).fit_transform(features)
        save_path = save_pth+"inference_perplexity_"+str(i)+".png"

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

def change_labels(df):
    # 0:normal(else) / 1:secretion / 2: resistance / 3: toxin / 4: anti-toxin / 5:virulence / 6: invasion / 7: kill / 8:phage / 9: putative / 10: hypothetical, unknown, uncharacterized
    # for fungi -:normal / 1:adhesion / 2: stress / 3: acquisition / 4: AMR
    labels = []
    # for item in df["max label"]:
    #     if item == "0":
    #         labels.append("normal")
    #     elif item == "1":
    #         labels.append("secretion")
    #     elif item == "2":
    #         labels.append("resistance")
    #     elif item == "3":
    #         labels.append("toxin")
    #     elif item == "4":
    #         labels.append("anti-toxin")
    #     else:
    #         labels.append("virulence")
    for item in df["max label"]:
        if item == "0":
            labels.append("normal")
        elif item == "1":
            labels.append("adhesion")
        elif item == "2":
            labels.append("stress")
        elif item == "3":
            labels.append("acquisition")
        else:
            labels.append("AMR")
    return labels

def change_col_name(df):
#     df.rename(columns={
#     0: 'normal',
#     1: 'secretion',
#     2: 'resistance',
#     3: 'toxin',
#     4: 'anti-toxin',
#     5: 'virulence'
# }, inplace=True)
    
    df.rename(columns={
    0: 'normal',
    1: 'adhesion',
    2: 'stress',
    3: 'acquisition',
    4: 'AMR'
}, inplace=True)
    
    return df

if __name__ == "__main__":
    print("input path: ", args.data_dir)
    print("output path: ", args.out_csv_dir)
    print("layers froze: ", args.freeze_layer)
    print("classes to train: ", args.n_classes)
    print("batch size: ", args.batch_size)
    print("learning rate: ", args.learning_rate)
    # initialize device - cuda if available cpu if else
    device = torch.device('cuda')
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # initialize model and tokenizer and device
    model = BertModel.from_pretrained("Rostlab/prot_bert_bfd", output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
    
    
    # # added lora
    config = LoraConfig(
        inference_mode=True, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value", "key"]
    )
    model = get_peft_model(model, config)
    # model = freeze_layers(model)
    
    print_trainable_parameters(model)

    model = CustomProtBERTModel(model, args.n_classes)


    print_trainable_parameters(model)
    print(model)

    
    # for name, param in model.named_parameters():
    #     print(f'{name}: {param.requires_grad}')
    # main_inference_path = "results/bacteria_inference/"
    if args.fine_tune == True:
        print("Fine Tune Model")
        print("Loading Best Weights ... ")
        # enter the path to use
        # best_weights_path = main_inference_path+"best_weights.pth"
        
        ## change weights path, change csv path for inference
        best_weights_path = "sungyoon/env_project/results/fungi_train/Swissprot_0725_3000/best_weights.pth"
        # test_csv = "Salmonella_typhimurium_(strainLT2)"

        ###BACTERIA NAME###
        for test_csv in os.listdir(main_train_csv_folder_path):
            # test_csv = "LT2-genome-total"
            print(test_csv.split(".")[0])
            test_csv = test_csv.split(".")[0]
            ##### CHANGE HERE ######
            model = load_model(model, best_weights_path)
            model.to(device)
            print(device)
        # else:
        #     print("Training From Pretrained Weights")
        #     print_trainable_parameters(model)
        #     model.to(device)
        #     print(device)



            # If train, load all dataset
            # If inference load inference data
            if args.mode == "inference":
                now = datetime.datetime.now()
                date = now.strftime('%m%d')
            
                print(device)
                print("Inference Mode")
                dataset_dir = main_train_csv_folder_path
                save_dir = main_output_save_folder_path
                infer_df = pd.read_csv(dataset_dir + test_csv+".csv")
                
                infer_df["sequence"] = infer_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
                infer_df['sequence'] = infer_df.apply(lambda row: " ".join(row["sequence"]), axis=1)
                infer_dataset = CustomProteinDataset(infer_df["sequence"], infer_df["label"], tokenizer, infer_df["tensor id"])
                infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

                infer_results = infer(model, infer_loader, save_dir)

                print(infer_results)
                infer_results.to_csv(save_dir + test_csv+date+".csv", index=False)
                df = pd.read_csv(dataset_dir + test_csv+".csv")
                
                df_compare = pd.read_csv(save_dir + test_csv +date+".csv")

                # df_compare.rename(columns={'0':'normal',
                #                    '1':'secretion',
                #                    '2':'resistance',
                #                    '3':'toxin',
                #                    '4':'anti-toxin',
                #                    '5':'virulence'}, inplace=True)

                # df_compare = df_compare.drop(['Unnamed: 0'], axis=1)
                df_compare_tensor_id = df_compare["tensor id"]
                df_compare = df_compare.drop(['tensor id'], axis=1)
                print(df_compare.head())
                print(df_compare.idxmax(axis=1))
                print(len(df_compare))
                df_compare = df_compare[(df_compare<0.8).any(axis=1)]
                print(len(df_compare))
                
                df_compare['max label'] = df_compare.idxmax(axis=1)
                df_compare["tensor id"] = df_compare_tensor_id
                labels_list = change_labels(df_compare)

                # print(df_compare.head())
                    
                df_new = merge_inferenece(infer_df, df_compare)
                print("1. Merged classification predictions with input data...")
                df_new["max label"] = labels_list
                df_new = change_col_name(df_new)
                print(df_new.head())
                check_labels_count(df_new)
                preds_check_labels_count(df_new)
                df_new.to_csv(save_dir+test_csv+date+".csv", index=False)
                # df = pd.read_csv(save_dir+"Salmonella_typhimurium_(strainLT2)_preds_"+date+".csv")
                # # print(df)
                # print("2. Label count of input data...")
                # check_labels_count(df)
                # print("3.Prediction count of input data...")
                # preds_check_labels_count(df)
                # df = matching_labels(df)
                
                # df_1 = df[df["matching label"] == True]
                # print("4. Label count where label matches predictions")
                # check_labels_count(df_1)
                # df_1.to_csv(save_dir+"batch_0_5class_only_virulence_toxin_preds_matching.csv", index=False)


            else:
                print("This is for INFERENCE CHECK TYPE")
        