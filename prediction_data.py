import pandas as pd
from sklearn import preprocessing
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
import os
from sklearn.model_selection import train_test_split
import sys, re
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

pd.set_option('display.max_columns', None)

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

    # df_keep = pd.concat([df, df_compare])
    # df_keep = df_keep.drop_duplicates(subset=['sequence'], keep=False)

    return result

def sample_trainable_df_5class(df, sample_count):
    df_by_labels = df.groupby(df.label)
    normal_df = df_by_labels.get_group(0)
    secretion_df = df_by_labels.get_group(1)
    resistant_df = df_by_labels.get_group(2)
    # toxin_df = df_by_labels.get_group(3)
    # anti_toxin_df = df_by_labels.get_group(4)
    # virulence_df= df_by_labels.get_group(5)/
    
    normal_random_df = normal_df.sample(sample_count)
    # secretion_random_df = secretion_df.sample(398)
    resistant_random_df = resistant_df.sample(sample_count)
    # toxin_random_df = toxin_df.sample(19)
    # anti_toxin_random_df = anti_toxin_df.sample(54)
    
    # sample_df = pd.concat([normal_random_df, secretion_random_df, resistant_random_df, toxin_random_df, anti_toxin_random_df, virulence_df])
    sample_df = pd.concat([normal_random_df, secretion_df, resistant_random_df])
    
    return sample_df

def add_preds_data(df, preds_df, sample_count):
    df_by_labels = preds_df.groupby(preds_df.label)
    secretion_df = df_by_labels.get_group(1)
    toxin_df = df_by_labels.get_group(3)
    anti_toxin_df = df_by_labels.get_group(4)
    virulence_df= df_by_labels.get_group(5)
    
    secretion_random_df = secretion_df.sample(2)
    virulence_random_df = virulence_df.sample(sample_count)
    toxin_random_df = toxin_df.sample(sample_count)
    anti_toxin_random_df = anti_toxin_df.sample(sample_count)
    
    sample_df = pd.concat([df, virulence_random_df, secretion_random_df, toxin_random_df, anti_toxin_random_df])
    
    return sample_df
    

def give_tensor_id(df):
    index_no = []
    for i in range(len(df["sequence"])):
       index_no.append(i)
    df["tensor id"] = index_no
    
    return df


# label bacteria data
# 0:normal(else) / 1:secretion / 2: resistance / 3: toxin / 4: anti-toxin / 5:virulence / 6: kill / 7: invasion / 8:phage / 9: effector / 10: putative / 11: hypothetical, unknown, uncharacterized
def label_data_bacteria(df):
    print("labelling bacteria sequences...\n-------")
    label_list = []
    for item in df["description"]:
        # print(item)
        if "uncha" in item:
            label_list.append(11)
        elif "unknown" in item:
            label_list.append(11)
        elif "hypothe" in item:
            label_list.append(11)
        elif "puta" in item:
            label_list.append(10)
        elif "effector" in item:
            label_list.append(9)
        elif "phage" in item:
            label_list.append(8)
        elif "inva" in item:
            # print("INVASION", item)
            label_list.append(7)
        elif "kill" in item:
            # print("KILL", item)
            label_list.append(6)
        elif "virul" in item:
                # print("VIRULENCE", item)
                label_list.append(5)
        elif 'toxi' in item:
            # print("TOXIN", item)
            if 'anti' in item:
                label_list.append(4)
            else:
                label_list.append(3)

        elif 'resi' in item:
            # print("RESISTANT", item)
            label_list.append(2)
        elif 'secre' in item:
            # print("SECRETION", item)
            label_list.append(1)
        else:
            label_list.append(0)


    df["label"] = label_list
    print("done labelling sequences!\n-------")
    return df


def batchable_data(df):
    index_no = []
    for i in range(len(df["id"])):
       index_no.append(i)
    df["tensor id"] = index_no
    df = label_data_bacteria(df)
    check_labels_count(df)
    df = df[df.label != 9]
    df = df[df.label != 8]
    df = df[df.label != 7]
    df = df[df.label != 6]
    print("Checking batched labels")
    check_labels_count(df)
    return df
    
if __name__ == "__main__":
    
    now = datetime.datetime.now()
    date = now.strftime('%m%d')
    num_added_data = 5200
    # main_train_csv_folder_path = "datasets/HMP_Swissprot_"+date+"_batch1_"+str(num_added_data)+"/"
    # main_output_save_folder_path = "results/HMP_Swissprot_"+date+"_batch1_"+str(num_added_data)+"/"
    main_train_csv_folder_path = "datasets/HMP_Swissprot_0528_batch1_5200/"
    main_output_save_folder_path = "results/HMP_Swissprot_0528_batch1_5200/"
    
    
    # 3 . Create New Dataset Part
    preds_save_pth = main_output_save_folder_path+"tremble_batch_results/"
    preds_data_pth = main_train_csv_folder_path
    save_new_pth = "datasets/HMP_Swissprot_"+date+"_batch1_"+str(num_added_data+400)+"/"
    results_save_new_pth = "results/HMP_Swissprot_"+date+"_batch1_"+str(num_added_data+400)+"/"
    if not os.path.exists(results_save_new_pth):
        os.makedirs(save_new_pth)
        os.makedirs(results_save_new_pth)
        os.makedirs(results_save_new_pth+"results_visualization")
        os.makedirs(results_save_new_pth+"tremble_batch_results")
        
    df = pd.read_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_1_5class_only_virulence_toxin_anti_toxin_secretion.csv")
    
    df_compare = pd.read_csv(preds_save_pth+"batch_1_5class_only_virulence_toxin_anti_toxin_secretion.csv")
    print("FUCK")
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
        
    df_new = merge_inferenece(df, df_compare)
    print("1. Merged classification predictions with input data...")
    print(df_new.head())
    check_labels_count(df_new)
    df_new.to_csv(preds_save_pth+"batch_1_5class_only_virulence_toxin_anti_toxin_preds_secretion.csv", index=False)
    df = pd.read_csv(preds_save_pth+"batch_1_5class_only_virulence_toxin_anti_toxin_preds_secretion.csv")
    # print(df)
    print("2. Label count of input data...")
    check_labels_count(df)
    print("3.Prediction count of input data...")
    preds_check_labels_count(df)
    df = matching_labels(df)
    
    df_1 = df[df["matching label"] == True]
    print("4. Label count where label matches predictions")
    check_labels_count(df_1)
    df_1.to_csv(preds_save_pth+"batch_1_5class_only_virulence_toxin_anti_toxin_preds_matching_secretion.csv", index=False)
    
    df_preds_matching = pd.read_csv(preds_save_pth+"batch_1_5class_only_virulence_toxin_anti_toxin_preds_matching_secretion.csv")
    df_SH = pd.read_csv("datasets/new_HMP_Swissprot/HMP_Swissprot_0403.csv")
    # check_labels_count(df_SH)
    df_0509 = join_train_data(preds_data_pth)
    
    print(df_0509)
    print("5. Label count of input trainable data(train, valid, test)")
    check_labels_count(df_0509)

    df_trainable = drop_duplicates(df_SH, df_0509)
    print("6. Label count where label matches predictions")
    check_labels_count(df_preds_matching)
    df_preds_matching = drop_duplicates(df_preds_matching, df_0509)
    print("7. Label count where label matches predictions and where sequences are not included in trainable data")
    check_labels_count(df_preds_matching)
    print("8. Label count where label matches predictions and where sequences are not included in Swissprot and HMP")
    check_labels_count(df_trainable)
    
    print("9. Label count of sampled data from swissprot and HMP")
    sampled_trainable_df = sample_trainable_df_5class(df_trainable, 400)
    check_labels_count(sampled_trainable_df)
    
    # print(sampled_trainable_df[sampled_trainable_df["label"] == 5])
    print("10. Label count of sampled data from prediction data")
    check_labels_count(df_preds_matching)
    sampled_trainable_df_w_preds = add_preds_data(sampled_trainable_df, df_preds_matching, 400)
    check_labels_count(sampled_trainable_df_w_preds)
    print(sampled_trainable_df_w_preds)
    print("===as=dfas=adfs=adfs=ad=f=adaf=d")
    add_train_data(path=preds_data_pth,save_path=save_new_pth, sample_df=sampled_trainable_df_w_preds)
        
  
    
    # # save batch 1 virulence and toxin only
    # df = pd.read_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_1.csv")
    # check_labels_count(df)
    # df_1 = batchable_data(df)
    # virulence_df = df_1[df_1["label"]==5]
    # anti_toxin_df = df_1[df_1["label"]==4]
    # toxin_df = df_1[df_1["label"]==3]
    # secretion_df = df_1[df_1["label"]==1]
    
    # join_df = pd.concat([virulence_df,anti_toxin_df, toxin_df, secretion_df], ignore_index = True)
    # check_labels_count(virulence_df)
    # check_labels_count(anti_toxin_df)
    # check_labels_count(toxin_df)
    # check_labels_count(secretion_df)
    # join_df.to_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_1_5class_only_virulence_toxin_anti_toxin_secretion.csv", index=False)



    # df_kill = df[df["label"] == 5]
    # print(df_kill)
    # df_kill.to_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0_5class_preds_virulence.csv", index=False)
    
    # df2 = df[['sequence', 'max label', 'tensor id']]
    # df2.rename(columns = {'max label' : 'label'}, inplace = True)
    # df1 = pd.read_csv("datasets/HMP_Swissprot_0430/train.csv")
    # check_labels_count(df1)
    # print(df2)
    # df2.to_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0_5class_preds_trainable.csv", index=False)
    # df2 = pd.read_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0_5class_preds_trainable.csv")
    # print(len(df1))
    # print(len(df2))
    # df3 = pd.concat([df1, df2], ignore_index = True)
    # print(len(df3))    
    # df3.to_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0_5class_preds_trainable_with_prev_train.csv", index=False)
    # df = pd.read_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0/batch_0_5class_preds_trainable.csv")
    # split_df = np.array_split(df, 10)
    # print(len(df))
    # for i, part in enumerate(split_df):
    #     print(f"Part {i}:")
    #     # print(part)
    #     part.to_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0/batch_0_5class_preds_trainable_batch_"+str(i)+".csv", index=False)
    #     print(len(part))
    
    # df = pd.read_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0/batch_0_5class_preds_trainable_batch_0.csv")
    # check_labels_count(df)
    # df1 = pd.read_csv("datasets/HMP_Swissprot_0430/train_original.csv")
    # df3 = pd.concat([df1, df], ignore_index = True)
    # print(len(df3))    
    # df3.to_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0/batch_0_5class_preds_trainable_batch_0_with_prev_train.csv", index=False)
    # df = pd.read_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0/batch_0_5class_preds_trainable_batch_0_with_prev_train.csv")
    # check_labels_count(df)
    
    # df = pd.read_csv("datasets/new_HMP_Swissprot/HMP_trainable.csv")
    # check_labels_count(df)