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
import logomaker
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
parser.add_argument('--n_classes', '-n', default=6, help='number of classes', type=int)
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

main_train_csv_folder_path = "datasets/Uniprot_1017_8800/"
main_output_save_folder_path = "results/bacteria_inference/"

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
    
def dataset_split(df, save_dir):
    train_set, test_set = train_test_split(df, test_size=0.3, random_state=42, stratify=
    df["label"].values.tolist())
    val_set, test_set = train_test_split(test_set, test_size=0.333, random_state=42, stratify=
    test_set["label"].values.tolist())
    my_train = train_set[["sequence", "label", "tensor id"]]
    my_valid = val_set[["sequence", "label", "tensor id"]]
    my_test = test_set[["sequence", "label", "tensor id"]]

    my_train.to_csv(save_dir + "train.csv")
    check_labels_count(my_train)
    my_valid.to_csv(save_dir + "valid.csv")
    check_labels_count(my_valid)
    my_test.to_csv(save_dir + "test.csv")
    check_labels_count(my_test)
       
def join_train_data(path):
    df_train = pd.read_csv(path+"train.csv")
    df_valid = pd.read_csv(path+"valid.csv")
    df_test = pd.read_csv(path+"test.csv")
    df = pd.concat([df_train, df_valid, df_test])
    
    return df

def add_train_data(path, save_path, sample_df):
    df_train_ori = pd.read_csv(path+"train.csv")
    df_valid_ori = pd.read_csv(path+"valid.csv")
    df_test_ori = pd.read_csv(path+"test.csv")
    
    
    train_set, test_set = train_test_split(sample_df, test_size=0.3, random_state=42, stratify=
    sample_df["label"].values.tolist())
    val_set, test_set = train_test_split(test_set, test_size=0.333, random_state=42, stratify=
    test_set["label"].values.tolist())
    
    my_train = train_set[["sequence", "label", "tensor id"]]
    my_valid = val_set[["sequence", "label", "tensor id"]]
    my_test = test_set[["sequence", "label", "tensor id"]]

    my_train = give_tensor_id(pd.concat([df_train_ori, my_train]))
    my_valid = give_tensor_id(pd.concat([df_valid_ori, my_valid]))
    my_test = give_tensor_id(pd.concat([df_test_ori, my_test]))
    
    my_train.to_csv(save_path + "train.csv", index = False)
    check_labels_count(my_train)
    my_valid.to_csv(save_path + "valid.csv", index = False)
    check_labels_count(my_valid)
    my_test.to_csv(save_path + "test.csv", index = False)
    check_labels_count(my_test)
    
def drop_duplicates(df, df_compare):
    # Column to use for matching
    column_to_match = 'sequence'

    # Merge the DataFrames on the specified column
    merged = pd.merge(df, df_compare, on=column_to_match, how='inner')

    # Filter out the rows from df_b that are duplicates of rows in df_a based on the specified column
    result = df[~df[column_to_match].isin(merged[column_to_match])]


    return result

def sample_trainable_df_5class(df, sample_count):
    df_by_labels = df.groupby(df.label)
    normal_df = df_by_labels.get_group(0)
    secretion_df = df_by_labels.get_group(1)
    resistant_df = df_by_labels.get_group(2)
    # toxin_df = df_by_labels.get_group(3)
    anti_toxin_df = df_by_labels.get_group(4)
    # virulence_df= df_by_labels.get_group(5)/
    
    normal_random_df = normal_df.sample(sample_count)
    secretion_random_df = secretion_df.sample(sample_count)
    resistant_random_df = resistant_df.sample(sample_count)
    # toxin_random_df = toxin_df.sample(19)
    anti_toxin_random_df = anti_toxin_df.sample(sample_count)
    
    # sample_df = pd.concat([normal_random_df, secretion_random_df, resistant_random_df, toxin_random_df, anti_toxin_random_df, virulence_df])
    sample_df = pd.concat([normal_random_df, secretion_random_df, resistant_random_df, anti_toxin_random_df])
    
    return sample_df

def add_preds_data(df, preds_df, sample_count):
    df_by_labels = preds_df.groupby(preds_df.label)
    toxin_df = df_by_labels.get_group(3)
    virulence_df= df_by_labels.get_group(5)
    virulence_random_df = virulence_df.sample(sample_count)
    toxin_random_df = toxin_df.sample(sample_count)
    
    sample_df = pd.concat([df, virulence_random_df, toxin_random_df])
    
    return sample_df
    
def give_tensor_id(df):
    index_no = []
    for i in range(len(df["sequence"])):
       index_no.append(i)
    df["tensor id"] = index_no
    
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
    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "ids": ids, "sequence": embs}

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
   
### BELOW CODE RETURNS THE MOTIF OF LENGTH 20 AROUND THE HIGHEST ATTENTION SCORE OF THE SEQUENCE
def attn_motif_extract(sequence, df):
    df_sorted = df.iloc[df.sum(axis=1).argsort()[::-1]]
    highest_AA = int(df_sorted.index[0])
    if highest_AA <= 10:
        return sequence[0:20]
    elif highest_AA >= len(sequence)-10:
        return sequence[-20:]
    else:
        return sequence[highest_AA-10:highest_AA+10]

def write_fasta(sequences, file_name, ids):
    with open(file_name, 'w') as fasta_file:
        for i, sequence in enumerate(sequences, 1):
            fasta_file.write(f'>seq{ids}_attn_head{i}\n')
            fasta_file.write(f'{sequence}\n')
            
def plot_attn(attentions, sequence, preds, ids):
    sequence = tokenizer.convert_ids_to_tokens(sequence[0, :])[1:-1]
    sequence_str = ''.join(map(str, sequence))
    
    columns_aas = [
    'A', 'C', 'D', 'E', 
    'F', 'G', 'H', 'I', 
    'K', 'L', 'M', 'N', 
    'P', 'Q', 'R', 'S',
    'T', 'V', 'W', 'Y'
    ]
    layer_num = len(attentions)
    head_num = attentions[0].size()[1]
    # print("number of heads:", head_num)

    for layer in range(layer_num):
        if layer == 29:
            # print(f'Layer number: {layer}')
            attention = attentions[layer]
            # print(attention.shape)
            ### BELOW CODE FOR PRINTING EACH ATTENTION HEAD OUTPUTS
            # fig, axes = plt.subplots(16, sharex=True, figsize=(20, 2.5*head_num+1))
            total_df = pd.DataFrame(0, index = range(len(sequence)),columns=columns_aas)
            total_motif = []
            for head in range(head_num):
                # print(f"\nAttention Head {head + 1}")
                # logo_df = pd.DataFrame(0.0, index=range(sequence_length), columns=columns_aas)
                df_array = np.zeros((len(sequence), 20))
                attn = attention[0][head].detach()
                # print("shape of attn:", attn.shape)
                # Fill in the attention scores for the relevant amino acids at each position
                for i, aa in enumerate(sequence):
                    aa_ind = columns_aas.index(aa)
                    df_array[i][aa_ind] = attn.mean(dim=0)[i+1]
                df = pd.DataFrame(df_array, columns=columns_aas)
                # df.index = df.index + 1
                ### BELOW CODE FOR PRINTNG EACH ATTENTION HEAD OUTPUTS
                # nn_logo = logomaker.Logo(df, color_scheme='chemistry', figsize=(20, 2.5), ax=axes[head])
                motif = attn_motif_extract(sequence_str, df)
                total_motif.append(motif)
            # print("Total Motif:", total_motif)
            # plt.show()
            ### BELOW CODE FOR PRINTING TOTAL ATTENTION HEAD OUTPUTS(SUM OF ALL HEADS)
            sequence_num = ids.item()
            if int(preds)==1:
                write_fasta(total_motif, f"results/attention_visualized/secretion/{sequence_num}.fasta", sequence_num)
            elif int(preds)==2:
                write_fasta(total_motif, f"results/attention_visualized/resistance/{sequence_num}.fasta", sequence_num)
            elif int(preds)==3:
                write_fasta(total_motif, f"results/attention_visualized/toxin/{sequence_num}.fasta", sequence_num)
            elif int(preds)==4:
                write_fasta(total_motif, f"results/attention_visualized/anti-toxin/{sequence_num}.fasta", sequence_num)
            elif int(preds)==5:
                write_fasta(total_motif, f"results/attention_visualized/virulence/{sequence_num}.fasta", sequence_num)
            
            del total_motif, total_df, ids, preds
            
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

            logits, final_features, attentions = model(input_ids, attention_mask, True)
            _, preds = torch.max(logits, 1)
            

            plot_attn(attentions, input_ids, preds, ids)

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


    # acc, f1, mcc, auroc = eval_metrics(LABELS, PREDS, args.n_classes)
    # print("ACCURACY\t",acc.cpu().numpy()*100)
    # print("F1-SCORE\t",f1.cpu().numpy())
    # print("MCC\t",mcc.cpu().numpy())
    # print("AUROC\t",auroc.cpu().numpy())
    # plot output
    tsne_plot_inference(total_final_features.cpu().numpy(), total_labels.cpu().numpy(), save_pth)
    # average scores
#    avg_acc = np.mean(ACC)
#    avg_f1 = np.mean(F1)
#    avg_mcc = np.mean(MCC)
#    avg_auroc = np.mean(AUROC)

    df = pd.DataFrame(IDS.cpu().numpy(), columns=["tensor id"]).astype('int64')
    df2 = pd.DataFrame(PREDS.cpu().numpy())
    df3 = pd.concat([df, df2], axis=1)
    print(df3)

    return df3


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
    labels = []
    for item in df["max label"]:
        if item == "0":
            labels.append("normal")
        elif item == "1":
            labels.append("secretion")
        elif item == "2":
            labels.append("resistance")
        elif item == "3":
            labels.append("toxin")
        elif item == "4":
            labels.append("anti-toxin")
        else:
            labels.append("virulence")
    
    return labels


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
        best_weights_path = "results/Uniprot_1017_8800/best_weights.pth"
        test_csv = "uniprot_keywords"
        model = load_model(model, best_weights_path)
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
        now = datetime.datetime.now()
        date = now.strftime('%m%d')
    
        print(device)
        print("Inference Mode")
        dataset_dir = main_train_csv_folder_path
        save_dir = main_output_save_folder_path
        infer_df = pd.read_csv(dataset_dir + "train.csv")
        
        infer_df["sequence"] = infer_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z","X"]), "G", regex=True)
        # infer_df["sequence"] = infer_df["sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
        
        infer_df['sequence'] = infer_df.apply(lambda row: " ".join(row["sequence"]), axis=1)
        infer_dataset = CustomProteinDataset(infer_df["sequence"], infer_df["label"], tokenizer, infer_df["tensor id"])
        infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        infer_results = infer(model, infer_loader, save_dir)

        print(infer_results)
        infer_results.to_csv(save_dir + test_csv+date+".csv", index=False)
        df = pd.read_csv(dataset_dir + test_csv+".csv")
        
        df_compare = pd.read_csv(save_dir + test_csv +date+".csv")
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
        # print(df_compare.head())
            
        df_new = merge_inferenece(infer_df, df_compare)
        print("1. Merged classification predictions with input data...")
        labels_list = change_labels(df_new)
        df_new["max label"] = labels_list
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
       