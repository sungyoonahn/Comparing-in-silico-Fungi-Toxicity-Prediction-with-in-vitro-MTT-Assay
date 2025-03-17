import pandas as pd
import os
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)

def check_labels_count(df):
    print("checking count for each labels...\n-------")
    print(pd.Series(df["label"]).value_counts().reset_index())
def check_max_labels_count(df):
    print("checking count for each labels...\n-------")
    print(pd.Series(df["max label"]).value_counts().reset_index())
    # return pd.Series(df["label"]).value_counts().reset_index()

def dataset_split(df, save_dir):
    train_set, test_set = train_test_split(df, test_size=0.3, random_state=42, stratify=
    df["label"].values.tolist())
    
    val_set, test_set = train_test_split(test_set, test_size=0.333, random_state=42, stratify=
    test_set["label"].values.tolist())
    
    my_train = train_set[["sequence", "label", "tensor id"]]
    my_valid = val_set[["sequence", "label", "tensor id"]]
    my_test = test_set[["sequence", "label", "tensor id"]]

    my_train.to_csv(save_dir + "train.csv", index=False)
    check_labels_count(my_train)
    my_valid.to_csv(save_dir + "valid.csv", index=False)
    check_labels_count(my_valid)
    my_test.to_csv(save_dir + "test.csv", index=False)
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


def sample_trembl_df(df):
    df_by_labels = df.groupby(df.label)
    secretion_df = df_by_labels.get_group(1)
    resistant_df = df_by_labels.get_group(2)
    toxin_df = df_by_labels.get_group(3)
    anti_toxin_df = df_by_labels.get_group(4)
    virulence_df= df_by_labels.get_group(5)
    # sample_num = len(virulence_df)
    
    secretion_random_df = secretion_df.sample(55292)
    resistant_random_df = resistant_df.sample(56466)
    toxin_random_df = toxin_df.sample(56076)
    anti_toxin_random_df = anti_toxin_df.sample(58030)
    virulence_random_df = virulence_df.sample(58280)
    
    sample_df = pd.concat([secretion_random_df, resistant_random_df, toxin_random_df, anti_toxin_random_df, virulence_random_df])
    sample_df["label"].astype('int64')
    sample_df = sample_df.reset_index(drop=True)
    
    return sample_df

#drops rows longer than 1024
def drop_length(df):
    print("removing rows longer than 1024...\n-------")
    df1 = df[df["length of sequence"].astype(int)<=1024]
    print("removed rows longer than 1024!\n-------")
    return df1

# df = pd.read_csv("results/Uniprot_1017_8800/trembl_batch_results/batch_0_to_6_all_5class_matching.csv")
# df = pd.read_csv("results/Uniprot_1017_8800/trembl_batch_results/TREMBL_1008_5_class_trainable_classification_results_1017_8800.csv")
# check_labels_count(df)
# check_max_labels_count(df)

# df_swissprot = pd.read_csv("datasets/Swissprot/Swissprot_1008_6_class_trainable.csv")
# df_trembl = pd.read_csv("datasets/TREMBL(UNIPROT)/Bacteria/TREMBL_1008_5_class_trainable.csv")
# check_labels_count(df_swissprot)
# check_labels_count(df_trembl)
# df_swissprot["length of sequence"] = df_swissprot["sequence"].apply(len)
# df_trembl["length of sequence"] = df_trembl["sequence"].apply(len)

# df_swissprot = drop_length(df_swissprot)
# df_trembl = drop_length(df_trembl)
# check_labels_count(df_swissprot)
# check_labels_count(df_trembl)

# df_trembl_sampled = sample_trembl_df(df_trembl)
# check_labels_count(df_trembl_sampled)

# combined_df = pd.concat([df_swissprot, df_trembl_sampled], ignore_index=True)
# check_labels_count(combined_df)
# # Match label 0 item count to 58583
# label_0_df = combined_df[combined_df["label"] == 0]
# if len(label_0_df) > 58583:
#     label_0_df = label_0_df.sample(58583)
# elif len(label_0_df) < 58583:
#     additional_samples = label_0_df.sample(58583 - len(label_0_df), replace=True)
#     label_0_df = pd.concat([label_0_df, additional_samples])

# # Combine the adjusted label 0 dataframe with the rest of the data
# remaining_df = combined_df[combined_df["label"] != 0]
# final_combined_df = pd.concat([remaining_df, label_0_df], ignore_index=True)
# check_labels_count(final_combined_df)

# dataset_split(final_combined_df, "datasets/Swissprot_TrEMBL_0306/")


df = pd.read_csv("datasets/Swissprot_TrEMBL_0306/train.csv")
check_labels_count(df)
df = pd.read_csv("datasets/Swissprot_TrEMBL_0306/valid.csv")
check_labels_count(df)
df = pd.read_csv("datasets/Swissprot_TrEMBL_0306/test.csv")
check_labels_count(df)


# df = pd.read_csv("results/bacteria_inference/Streptococcus_mutans_swissprot1029_oldmodel_compare.csv")
# match_df = df[(df["matching"] == "Match") & (df["max label"] == "normal")]
# mismatch_df = df[(df["matching"] == "No match") & (df["max label"] == "normal")]
# print(len(match_df))
# print(len(mismatch_df))
# df1 = pd.read_csv("results/bacteria_inference/Salmonella_typhimurium_(strainLT2)0806_oldmodel.csv")
# df2 = pd.read_csv("results/bacteria_inference/Salmonella_typhimurium_(strainLT2)1029.csv")
# # old model - description
# check_labels_count(df1)
# len(df1)
# # new model - keyword
# check_labels_count(df2)
# len(df2)

# print(df1["max label"])
# print(df2["max label"])

# df_main = df2

# df_main["prev max label"] = df1["max label"]
# df_main["matching"] = (df_main["max label"] != df_main["prev max label"])
# df_main["matching"] = df_main["matching"].map({True: "No match", False: "Match"})
# print(df_main.head)

# df_main.to_csv("results/bacteria_inference/Salmonella_typhimurium_(strainLT2)1029_oldmodel_compare_0224_check.csv", index=False)
# # write code to compare the two dataframes max label column

# df1["prev max label"] = df2["max label"]

# df1["matching"] = df1["max label"] != df1["prev max label"]
# df1["matching"] = df1["matching"].map({True: "No match", False: "Match"})

# df1.to_csv("results/bacteria_inference/Streptococcus_mutans_swissprot1029_oldmodel_compare.csv", index=False)

# df = pd.read_csv("results/bacteria_inference/Streptococcus_mutans_swissprot1029_oldmodel_compare.csv")

# # count "match" in clomn
# print(df["matching"].value_counts())



# df = pd.read_csv("results/Uniprot_1017_0400/results.csv")

# results_path = "results/"
# file_list = os.listdir(results_path)
# # print(file_list)
# filtered_list = [item for item in file_list if "1017" in item]
# sorted_filtered_list = sorted(filtered_list)

# batch_best_losses = []
# best_ACCS = []
# y_values = []
# for item in sorted_filtered_list:
#     pth_list = os.listdir(results_path+item+"/")
#     y_values.append(int(item.split("_")[2]))
#     df = pd.read_csv(results_path+item+"/results.csv")
#     dfs = [df[i:i+4] for i in range(0, len(df), 4)]
#     # print(df.head())
    
#     filtered_list = [item for item in pth_list if "loss" in item]
#     best_losses = []
#     for loss in filtered_list:
#         best_loss = loss.split("_")[2].split(".")[1]
#         best_epoch = loss.split("_")[0]
#         best_losses.append(best_loss)
#     # get ACC of best loss testset
#     for df in dfs:
#         print(df.index.values[0])
#         if df["epoch"][df.index.values[0]] == best_epoch:
#             best_ACC = df["ACC"][df.index.values[3]]
#         else:
#             continue

#     print(best_ACC)
#     print(best_losses)
#     batch_best_losses.append(min(best_losses))
#     best_ACCS.append(best_ACC)
    
# import os
# import pandas as pd
# from Bio import SeqIO


# train_df = pd.read_csv("junk_data/train.csv")
# valid_df = pd.read_csv("junk_data/valid.csv")
# test_df = pd.read_csv("junk_data/test.csv")

# merged_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
# print(len(merged_df))

# # print(merged_df.head())
# print(merged_df[merged_df.duplicated(subset=["id"])])

# trembl_df = pd.read_csv("/media/prml/D/env_project/datasets_bacteria_own_keywords/new_HMP_Swissprot/Swissprot_trainable.csv")
# others_df = trembl_df[trembl_df["label"] != 0]
# others_df = others_df[others_df["label"] != 10]
# others_df = others_df[others_df["label"] != 9]
# others_df = others_df[others_df["label"] != 8]
# others_df = others_df[others_df["label"] != 7]
# others_df = others_df[others_df["label"] != 6]
# print(others_df[others_df.duplicated(subset=["id"])])
# print(trembl_df[trembl_df.duplicated(subset=["sequence"])])


# def check_duplicates(input_list):
#     duplicates = [item for item in set(input_list) if input_list.count(item) > 1]
#     return duplicates

# content_id =[]
# data_path = "datasets/Swissprot/"
# for record in SeqIO.parse(data_path+"uniprotkb_taxonomy_id_2_AND_keyword_KW_AMR_Swiss_10_07.fasta", "fasta"):
#     content_id.append(record.id)
# print(len(content_id))

# for record in SeqIO.parse(data_path+"uniprotkb_taxonomy_id_2_AND_keyword_KW_Secr_Swiss_10_07.fasta", "fasta"):
#     content_id.append(record.id)
# print(len(content_id))
    
# for record in SeqIO.parse(data_path+"uniprotkb_taxonomy_id_2_AND_keyword_KW_toxin_Swiss_10_07.fasta", "fasta"):
#     content_id.append(record.id)
# print(len(content_id))
    
# for record in SeqIO.parse(data_path+"uniprotkb_taxonomy_id_2_AND_keyword_KW_toxin_Swiss_10_07.fasta", "fasta"):
#     content_id.append(record.id)
# print(len(content_id))
    
# for record in SeqIO.parse(data_path+"uniprotkb_taxonomy_id_2_AND_keyword_KW_virul_Swiss_10_07.fasta", "fasta"):
#     content_id.append(record.id)
# print(len(content_id))
# # print(content_id)

# duplicates = check_duplicates(content_id)
# print(duplicates)
# print(len(duplicates))

# import os
# import pandas as pd 


# fungi_data_path = "datasets/fungi_inference/"
# for fungi_tsv in os.listdir(fungi_data_path):
#     csv_table = pd.read_table(fungi_data_path+fungi_tsv, sep='\t')
#     csv_table.to_csv(fungi_data_path+fungi_tsv.split(".")[0]+'.csv',index=False)

# bacteria_path = "env_project/results/bacteria_inference/Streptococcus_mutans_swissprot1029.csv"
# df = pd.read_csv(bacteria_path)
# check_labels_count(df)


# import os
# import pandas as pd

# root = "sungyoon/env_project/results/bacteria_inference/NCBI_bacteria_csv_0218/"
# for data in os.listdir(root):
#     csv_path = root+data
#     df = pd.read_csv(csv_path)
#     # df.rename(columns={'0':'normal',
#     #                     '1':'adhesion',
#     #                     '2':'stress',
#     #                     '3':'acquisition',
#     #                     '4':'AMR'}, inplace=True)
#     df.rename(columns={'0':'normal',
#                     '1':'secretion',
#                     '2':'resistance',
#                     '3':'toxin',
#                     '4':'anti-toxin',
#                     '5':'virulence'}, inplace=True)
#     print(data)
#     labels_df = check_labels_count(df)
#     print(df.head())
#     df2 = pd.concat([df, labels_df], axis=1, join='outer')
#     print(df2)
#     df2.to_csv(csv_path, index=False)