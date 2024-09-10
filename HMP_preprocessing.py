import pandas as pd
from sklearn import preprocessing
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
import os
from sklearn.model_selection import train_test_split
import sys, re
from Bio import SeqIO

pd.set_option('display.max_columns', None)
root_data_path = "datasets/"
save_file_path = "uniprot_fungi.csv"
import tqdm
bacteria_list = ["[Brevibacterium] frigoritolerans",
"Acinetobacter johnsonii",
"Acinetobacter lwoffii",
"Acinetobacter pittii",
"Agrococcus baldri",
"Brevibacterium senegalense",
"Arthrobacter bussei",
"Bacillus aerius",
"Priestia aryabhattai",
"Bacillus cereus",
"Paenibacillus macquariensis",
"Bacillus infantis",
"Bacillus licheniformis",
"Bacillus marisflavi",
"Priestia megaterium",
"Bacillus proteolyticus",
"Paenibacillus terrigena",
"Bacillus thuringiensis",
"Bacillus wiedmannii",
"Brachybacterium paraconglomeratum",
"Brevibacterium sediminis",
"Brevundimonas diminuta",
"Brevundimonas vesicularis",
"Mammaliicoccus sciuri",
"Chryseobacterium cucumeris",
"Chryseobacterium gwangjuense",
"Chryseobacterium taihuense",
"Corynebacterium doosanense",
"Corynebacterium glutamicum",
"Pseudarthrobacter sulfonivorans",
"Dermacoccus profundi",
"Psychrobacillus psychrodurans",
"Kocuria salsicia",
"Rhodococcus biphenylivorans",
"Exiguobacterium acetylicum",
"Paenarthrobacter nicotinovorans",
"Kocuria carniphila",
"Kocuria marina",
"Kocuria palustris",
"Kocuria rhizophila",
"Kocuria rosea",
"Leucobacter chromiiresistens",
"Lysinibacillus pakistanensis",
"Lysinibacillus parviboronicapiens",
"Planomicrobium okeanokoites",
"Massilia haematophila",
"Rosenbergiella epipactidis",
"Dietzia cercidiphylli",
"Microbacterium oxydans",
"Massilia agri",
"Microbacterium testaceum",
"Kocuria gwangalliensis",
"Micrococcus yunnanensis",
"Moraxella osloensis",
"Neomicrococcus lactis",
"Nocardia globerula",
"Paenibacillus bovis",
"Microbacterium arborescens",
"Brevundimonas bullata",
"Pseudomonas plecoglossicida",
"Staphylococcus edaphicus",
"Pseudarthrobacter oxydans",
"Pseudarthrobacter siccitolerans",
"Micrococcus aloeverae",
"Pseudomonas parafulva",
"Pseudomonas psychrotolerans",
"Pseudomonas putida",
"Pseudomonas straminea",
"Roseomonas mucosa",
"Streptomyces californicus",
"Salmonella typhimurium LT2"]

def remove_duplicates(df, organism_list):
    test_df = pd.DataFrame(columns=['bacteria species', 'id', 'description', 'sequence', 'length of sequence'])
    for organism_name in organism_list:
        matching_rows = df[df['description'].str.contains(organism_name, case=False, na=False)]
        test_df = pd.concat([test_df, matching_rows])

    train_df = pd.concat([df, test_df])

    train_df = train_df.drop_duplicates(keep=False)

    return train_df, test_df


def check_labels_count(df):
    print(pd.Series(df["label"]).value_counts().reset_index())


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



def tensor_id_merge(df1, df2):
    df3 = pd.merge(df1, df2, on='tensor id', how='inner')

    return df3

def groupby_labels(df):
    # print(pd.Series(df["label"]).value_counts().reset_index())
    # df_by_labels = df.groupby(df.label)
    # # kill_df = df_by_labels.get_group(6)
    # # invasion_df = df_by_labels.get_group(5)
    # virulence_df = df_by_labels.get_group(2)
    # toxin_df = df_by_labels.get_group(1)
    # # resistance_df = df_by_labels.get_group(1)
    # # secretion_df = df_by_labels.get_group(1)
    # normal_df = df_by_labels.get_group(0)
    # # #
    # # # # #randomly selected data from normal data
    # normal_random_df = normal_df.sample(4609)
    # # resistance_random_df = resistance_df.sample(4255)
    # # secretion_random_df = secretion_df.sample(4255)
    # virulence_df = virulence_df.sample(4609)
    # toxin_random_df = toxin_df.sample(4609)
    # # final_df = pd.concat([kill_df, invasion_df, virulence_df, toxin_df, resistance_df, secretion_df, normal_random_df])
    # # print(final_df)
    # #  print(pd.Series(final_df["label"]).value_counts().reset_index())
    return df

def label_data(df):
    label_list = []
    for item in df["description"]:
        # print(item)
        if "uncha" in item:
            label_list.append(7)
        elif "unknown" in item:
            label_list.append(7)
        elif "hypothe" in item:
            label_list.append(7)
            
        elif "kill" in item:
            # print("KILL", item)
            label_list.append(6)
        elif "inva" in item:
            # print("INVASION", item)
            label_list.append(5)
        elif "virul" in item:
            # print("VIRULENCE", item)
            label_list.append(4)

        elif 'toxi' in item:
            # print("TOXIN", item)
            if 'anti' in item:
                label_list.append(0)
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

    return df

def label_fungi_data(df):
    label_list = []
    for item in df["description"]:
        # print(item)
        if "hypothetical" in item:
            # print("unknown", item)
            label_list.append(3)
        elif "unknown" in item:
            label_list.append(3)
        elif "virul" in item:
            # print("VIRULENCE", item)
            label_list.append(2)
        elif 'toxi' in item:
            # print("TOXIN", item)
            if 'anti' in item:
                label_list.append(0)
            else:
                label_list.append(1)
        else:
            label_list.append(0)


    df["label"] = label_list

    return df


def group_by_class(df):
    df_by_labels = df.groupby(df.label)
    unknown_df = df_by_labels.get_group(7)
    kill_df = df_by_labels.get_group(6)
    invasion_df = df_by_labels.get_group(5)
    virulence_df = df_by_labels.get_group(4)
    toxin_df = df_by_labels.get_group(3)
    resistance_df = df_by_labels.get_group(2)
    secretion_df = df_by_labels.get_group(1)
    normal_df = df_by_labels.get_group(0)

    final_all = pd.concat([kill_df, invasion_df, virulence_df, toxin_df, resistance_df, secretion_df, normal_df])
    final_4class = pd.concat([secretion_df, resistance_df, toxin_df, normal_df])
    return unknown_df, final_all, final_4class

def balance_data(df):
    df_by_labels = df.groupby(df.label)
    kill_df = df_by_labels.get_group(6)
    invasion_df = df_by_labels.get_group(5)
    virulence_df = df_by_labels.get_group(4)
    toxin_df = df_by_labels.get_group(3)
    resistance_df = df_by_labels.get_group(2)
    secretion_df = df_by_labels.get_group(1)
    normal_df = df_by_labels.get_group(0)

    normal_random_df = normal_df.sample(10860)
    final_all = pd.concat([kill_df, invasion_df, virulence_df, toxin_df, resistance_df, secretion_df, normal_random_df])
    final_4class = pd.concat([secretion_df, resistance_df, toxin_df, normal_random_df])
    return final_all, final_4class



def group_balance_fungi_data(df):
    df_by_labels = df.groupby(df.label)
    virulence_df = df_by_labels.get_group(2)
    toxin_df = df_by_labels.get_group(1)
    normal_df = df_by_labels.get_group(0)

    normal_random_df = normal_df.sample(11828)

    sample_df = pd.concat([normal_random_df, toxin_df, virulence_df])
    return sample_df



df = pd.read_csv("datasets/new_HMP/HMP_new_1024_ALL.csv")
check_labels_count(df)

df = pd.read_csv("datasets/new_HMP/4_class/train.csv")
check_labels_count(df)
df = pd.read_csv("datasets/new_HMP/4_class/valid.csv")
check_labels_count(df)
df = pd.read_csv("datasets/new_HMP/4_class/test.csv")
check_labels_count(df)
# df = pd.read_csv("datasets/fungi_32_species/original_data/fungi_1024_3class_32organism_train.csv")
# df.rename(columns = {'fungi species':'species'}, inplace = True)
# check_labels_count(df)
# df1 = label_fungi_data(df)
# check_labels_count(df1)

# hypothetical_fungi_df = df1[df1["label"] == 3]
# df2 = df1[df1["label"] != 3]

# df3 = group_balance_fungi_data(df2)

# index_no = []
# for i in range(len(df3["id"])):
#     index_no.append(i)
# df3["tensor id"] = index_no

# check_labels_count(df3)
# print(df3)
# df3.to_csv("datasets/fungi_32_species_new/somehowbalanced.csv")
# dataset_split(df3, "datasets/fungi_32_species_new/")





# df = pd.read_csv("datasets/HMP/HMP_1024_ALL.csv")
# check_labels_count(df)
# df = pd.read_csv("datasets/new_uniprot_bacteria/70family.csv")
# df = label_data(df)
#
# index_no = []
# for i in range(len(df["id"])):
#     index_no.append(i)
# df["tensor id"] = index_no
#
# print("label count for inference data")
# check_labels_count(df)
# print("length of total data")
# print(len(df))
#
# print("number of unique bacteria species")
# print(len(pd.unique(df['species'])))
# print(pd.unique(df['species']))
#
# df.to_csv("datasets/new_uniprot_bacteria/70family.csv", index=False)
# #
# df1 = df[df["label"] != 0]
# print(df1)
# check_labels_count(df1)
# df1["label"].replace(4, 0, inplace=True)
# df1["label"].replace(5, 0, inplace=True)
# df1["label"].replace(6, 0, inplace=True)
# df1["label"].replace(7, 0, inplace=True)
#
# check_labels_count(df1)
#
# print(df1)
# df1.to_csv("datasets/new_uniprot_bacteria/70family_no_normal_inference_only.csv", index=False)
# check_labels_count(df1)

# df = pd.read_csv("datasets/new_uniprot_bacteria/70family_no_normal_inference_only.csv")
# df1 = df[["sequence", "label", "tensor id"]]
# df1.to_csv("datasets/new_uniprot_bacteria/70family_no_normal_inference_only.csv", index=False)
#
# print(df1)

def tensor_id_merge_bacteria(df1, df2):
    
    df3 = pd.merge(df1, df2, on='tensor id', how='inner')
    df3["label"] = df3["label"].replace(0, "normal")
    df3["label"] = df3["label"].replace(1, "secretion")
    df3["label"] = df3["label"].replace(2, "resistance")
    df3["label"] = df3["label"].replace(3, "toxin")
    df3 = df3.drop('Unnamed: 0', axis=1)   

    return df3

# df = pd.read_csv("datasets/bacteria_inference/Salmonella_typhimurium_(strainLT2).csv")
# df_compare = pd.read_csv("results/Salmonella_typhimurium_(strainLT2).csv")

# df_new = tensor_id_merge_bacteria(df, df_compare)
# print(df_new)

# df_new.to_csv("results/Salmonella_typhimurium_(strainLT2)_results.csv", index=False)
def drop_unnecessary_columns(df):
    
    df = df.drop('sequence', axis=1)
    df = df.drop('length of sequence', axis=1)
    df1 = df.drop('id', axis=1)
    return df1


# df = pd.read_csv("results/bacteria_inference/Salmonella_typhimurium_(strainLT2)_results.csv")

# df_normal = df[df["label"] == "normal"]

# df_possible_toxin = df_normal[df_normal["toxin"] > 0.3]
# df_check = df_possible_toxin[df_possible_toxin["resistance"]>0.3]

# #df_possible_resistance = df_normal[df_normal["resistance"] > 0.5]
# #df_possible_secretion = df_normal[df_normal["secretion"] > 0.5]

# print(len(df_check))


#print(len(df_possible_resistance))
#print(len(df_possible_secretion))

#df_possible_toxin_top = df_possible_toxin.sort_values("toxin", ascending=False).round(4)
#df_possible_resistance_top = df_possible_resistance.sort_values("resistance", ascending=False).round(4)
#df_possible_secretion_top = df_possible_secretion.sort_values("secretion", ascending=False).round(4)

#df_possible_toxin = drop_unnecessary_columns(df_possible_toxin)
#df_possible_resistance_top = drop_unnecessary_columns(df_possible_resistance_top)
#df_possible_secretion_top = drop_unnecessary_columns(df_possible_secretion_top)

#df_possible_toxin_top.to_csv("results/bacteria_inference/possible/Salmonella_typhimurium_(strainLT2)_possible_toxin.csv", index=False)
#df_possible_resistance_top.to_csv("results/bacteria_inference/possible/Salmonella_typhimurium_(strainLT2)_possible_resistance.csv", index=False)
#df_possible_secretion_top.to_csv("results/bacteria_inference/possible/Salmonella_typhimurium_(strainLT2)_possible_secretion.csv", index=False)



# df1 = pd.read_csv("results/70family_no_normal_inference_only.csv")
# df2 = pd.read_csv("datasets/bacteria_inference/70family.csv")



# df_new = tensor_id_merge_bacteria(df1, df2)
# print(df_new)



# df2 = pd.read_csv("results/70family_no_normal_inference_only_results.csv")
# df2["label"] = df2["label"].replace(1, "secretion")
# df2["label"] = df2["label"].replace(2, "resistance")
# df2["label"] = df2["label"].replace(3, "toxin")
#
# # print(df2[df2["label"]=="virulence"])
# # df_new = df2.iloc[:,[3,4,5,6,7,8,9,10,11,12]]
# # df2.to_csv("datasets/fungi_4609/fungi_1024_3class_30organisms_test_no_normal_class_inference_results.csv", index=False)
#
# # print(df2)
# secretion_df = df2[df2["label"] == "secretion"]
# resistance_df = df2[df2["label"] == "resistance"]
# toxin_df = df2[df2["label"] == "toxin"]
#
# secretion_top = secretion_df.sort_values("secretion", ascending=False).head(5).round(4)
# resistance_top = resistance_df.sort_values("resistance", ascending=False).head(5).round(4)
# toxin_top = toxin_df.sort_values("toxin", ascending=False).head(5).round(4)
#
#
# # top_10 = pd.concat([secretion_top])
# top_10 = pd.concat([secretion_top, resistance_top, toxin_top])
#
# print(top_10)
# top_10.to_csv("results/70family_no_normal_inference_only_results_top5.csv", index=False)







# new_label_list = []
# for item in df["label"]:
#     new_label_list.append(int(item[1]))
#
# print("TOTAL DATA")
# df["label"] = new_label_list
# df["label"] = df["label"].astype("int64")
# df['tensor id'] = df['tensor id'].astype('int64')
# df = label_data(df)
# # print(df.head())
# print("label count for total data")
# check_labels_count(df)
# print("length of total data")
# print(len(df))
#
# print("TRAIN DATA")
# train_df, test_df = remove_duplicates(df, bacteria_list)
# train_df['tensor id'] = train_df['tensor id'].astype('int64')
# train_df["label"] = train_df["label"].astype("int64")
#
# print("label count for total trainable data")
# check_labels_count(train_df)
# print("length of total trainable data")
# print(len(train_df))
# print("number of unique bacteria species")
# print(len(pd.unique(train_df['bacteria species'])))
#
# print("TEST DATA")
# test_df['tensor id'] = test_df['tensor id'].astype('int64')
# test_df["label"] = test_df["label"].astype("int64")
# print("label count for test data(one of 71 bacteria species)")
# check_labels_count(test_df)
# print("length of total test data in the HMP dataset")
# print(len(test_df))
# print("number of unique bacteria species in the HMP test set")
# print(len(pd.unique(test_df['bacteria species'])))
#
# # print("label count for test data(one of 71 bacteria species)")
# check_labels_count(train_df)
# unknown_df, final_all, final_4class = group_by_class(train_df)
# print(final_all)
# check_labels_count(final_all)
#
# save_folder_path = "datasets/new_HMP/"
# unknown_df.to_csv(save_folder_path+"HMP_new_1024_unknown.csv", index=False)
# final_all.to_csv(save_folder_path+"HMP_new_1024_ALL.csv", index=False)
# final_4class.to_csv(save_folder_path+"HMP_new_1024_4class.csv", index=False)
#
#
# final_all_balanced, final_4class_balanced = balance_data(final_all)
# print(len(final_4class_balanced))
# check_labels_count(final_4class_balanced)
# final_all_balanced.to_csv(save_folder_path+"HMP_new_1024_ALL_balanced.csv", index=False)
# final_4class_balanced.to_csv(save_folder_path+"HMP_new_1024_4class_balanced.csv", index=False)
#
# final_4class_balanced['tensor id'] = final_4class_balanced['tensor id'].astype('int64')
# final_4class_balanced["label"] = final_4class_balanced["label"].astype("int64")
# dataset_split(final_4class_balanced, save_folder_path+"10860/")
#
# test_df.to_csv(save_folder_path+"HMP_71_species.csv", index=False)



# df = pd.read_csv("datasets/uniprot_bacteria_1224.csv")
# print(df)

# df.rename(columns = {'bacteria species' : 'bacteria species and strand'}, inplace = True)
# print(df)
# fungi_species_list = []
# for fungi_species in df["bacteria species and strand"]:
#     split_string = str(fungi_species).split(" ")
#     if len(split_string) == 1:
#         fungi_species_list.append("nan")
#     else:
#         combined_string = split_string[0] + " " + split_string[1]
#         fungi_species_list.append(combined_string)

# df["bacteria species"] = fungi_species_list
# print("before removing proteins with only family")
# print(len(df))
# bacteria_df_nan = df[df["bacteria species"] != "nan"]
# print("after removing proteins with only family")
# print(len(bacteria_df_nan))
# print("after removing proteins longer than 1024")
# print(len(bacteria_df_nan))
# bacteria_df_1024 = bacteria_df_nan.loc[bacteria_df_nan['length of sequence']<=1024]


# df2 = label_data(bacteria_df_1024)
# check_labels_count(df2)
# index_no = []
# for i in range(len(df2["id"])):
#    index_no.append(i+1890358)
# df2["tensor id"] = index_no

# print(df2)
# df2.to_csv("uniprot_new_1024_ALL_1224.csv", index = False)
# df = pd.read_csv("datasets/new_HMP_Swissprot/4_class/train.csv")
# check_labels_count(df)
# print(df)
# df1 = pd.read_csv("datasets/new_HMP_Swissprot/swissprot_new_1024_ALL_1224.csv")
# check_labels_count(df1)
# df2 = df1[["sequence", "label", "tensor id"]]

# df_by_labels = df2.groupby(df2.label)
# check_labels_count(df2)

# toxin_df = df_by_labels.get_group(3)
# resistance_df = df_by_labels.get_group(2)
# secretion_df = df_by_labels.get_group(1)
# normal_df = df_by_labels.get_group(0)
# # # # # # #randomly selected data from normal data
# normal_random_df = normal_df.sample(968)
# # # # resistance_random_df = resistance_df.sample(4255)
# # # # secretion_random_df = secretion_df.sample(4255)
# # # virulence_df = virulence_df.sample(4609)
# # # toxin_random_df = toxin_df.sample(4609)
# final_df = pd.concat([toxin_df, resistance_df, secretion_df, normal_random_df])
# print(final_df)
# check_labels_count(final_df)

# df3 = pd.concat([df, final_df])
# print(df3)
# check_labels_count(df3)
# df3.to_csv("datasets/new_HMP_Swissprot/4_class/train.csv", index = False)
# train_df, test_df = remove_duplicates(df2, bacteria_list)
# train_df = train_df.drop(columns = ['tensor id'])
# print(len(train_df))
# print(train_df)
# check_labels_count(train_df)


#
# bacteria_df_1024 = bacteria_df_1024.iloc[:,[4,5,6,7,8,9,10,11,12]]

# bacteria_df_1024.to_csv("datasets/HMP/HMP_1024_ALL.csv", index=False)

# df = pd.read_csv("datasets/HMP/HMP_1024_ALL.csv")
# check_labels_count(df)

# df = pd.read_csv("datasets/HMP/4_class/train.csv")
# check_labels_count(df)
# df = pd.read_csv("datasets/HMP/4_class/valid.csv")
# check_labels_count(df)
#
# df = pd.read_csv("datasets/HMP/4_class/test.csv")
# check_labels_count(df)

# df = pd.read_csv("datasets/HMP/HMP_1024_ALL.csv")
# # # print(df)
# new_label_list = []
# for item in df["label"]:
#     new_label_list.append(int(item[1]))
#
# df["label"] = new_label_list
# check_labels_count(df)
# df_by_labels = df.groupby(df.label)
# check_labels_count(df)
#
# kill_df = df_by_labels.get_group(6)
# invasion_df = df_by_labels.get_group(5)
# virulence_df = df_by_labels.get_group(4)
# toxin_df = df_by_labels.get_group(3)
# resistance_df = df_by_labels.get_group(2)
# secretion_df = df_by_labels.get_group(1)
# normal_df = df_by_labels.get_group(0)
# #
# # # # #
# # # # # # #randomly selected data from normal data
# normal_random_df = normal_df.sample(11929)
# # # # resistance_random_df = resistance_df.sample(4255)
# # # # secretion_random_df = secretion_df.sample(4255)
# # # virulence_df = virulence_df.sample(4609)
# # # toxin_random_df = toxin_df.sample(4609)
# final_df = pd.concat([toxin_df, resistance_df, secretion_df, normal_random_df])
# print(final_df)
# final_df.to_csv("datasets/HMP/HMP_1024_normal_11929_toxin_resistance_secretion.csv", index=False)
# dataset_split(final_df, "datasets/HMP/4_class/")
# check_labels_count(final_df)
# #
# df = pd.read_csv("datasets/HMP/HMP_1024_normal_11929_rest_ALL.csv")
# dataset_split(df, "datasets/HMP/11929/")


# # ### change file extention from txt to fasta ###
# # # import glob
# # # files = glob.glob("HMP_protein_data/Airways/*.txt")
# # # for name in files:
# # #     if not os.path.isdir(name):
# # #         src = os.path.splitext(name)
# # #         os.rename(name, src[0]+'.fa')
# #
# #
# #
# #
# # # first take fa files and store them in to csv format
# # # csv table columns include, target organ, bacteria species, id, description, sequence, length of sequence,wprk kllkj
# #
# maindf= pd.DataFrame(columns=['target organ', 'bacteria species', 'id', 'description', 'sequence', 'length of sequence'])
#
# maindf = pd.DataFrame(columns=['fungi_species', 'id', 'description', 'sequence', 'length of sequence'])
#
# fungi_sequences = SeqIO.parse(open("datasets/uniprot_bacteria_1224.fasta"), "fasta")

# fungi_name_list = []
# id_list = []
# seq_list = []
# des_list = []
# seq_len_list = []

# for fasta in fungi_sequences:

#     id_list.append(fasta.id)
#     seq_list.append(str(fasta.seq))
#     des_list.append(fasta.description)
#     seq_len_list.append(len(fasta.seq))
#     match = re.search(r'OS=(.*?) OX=', fasta.description)
#     fungi_name_list.append(match.group(1))
#     print(fasta.id)

# df = pd.DataFrame([x for x in zip(fungi_name_list,
#                                               id_list, des_list, seq_list, seq_len_list)],
#                               columns=['bacteria species', 'id', 'description',
#                                        'sequence', 'length of sequence'])

# df.to_csv("datasets/uniprot_bacteria_1224.csv", index=False)
#
# maindf = pd.concat([maindf, df], axis=0)
#
# maindf.to_csv(root_data_path+save_file_path)
#
# for item in os.listdir(root_data_path):
#     print(item)
#     print("Name of organ: ", item)
#     file_path = root_data_path+item+"/"
#     for file in os.listdir(file_path):
#         bacteria_name = file.split(".")[0]
#         print(bacteria_name)
#         target_organ_list = []
#         bacteria_name_list = []
#         id_list = []
#         seq_list = []
#         des_list = []
#         seq_len_list = []
#
#         fasta_sequences = SeqIO.parse(open(file_path+file), "fasta")
#         for fasta in fasta_sequences:
#             target_organ_list.append(item)
#             bacteria_name_list.append(bacteria_name)
#             id_list.append(fasta.id)
#             seq_list.append(str(fasta.seq))
#             des_list.append(fasta.description)
#             seq_len_list.append(len(fasta.seq))
#
#
#             df = pd.DataFrame([x for x in zip(target_organ_list, bacteria_name_list,
#                                               id_list, des_list, seq_list, seq_len_list)],
#                               columns=['target organ', 'bacteria species', 'id', 'description',
#                                        'sequence', 'length of sequence'])
#         maindf = pd.concat([maindf, df], axis=0)
# #
#
# maindf.to_csv(save_file_path)
# #
# label_list = []
#
# fungi_df = pd.read_csv(root_data_path+save_file_path)
# print(fungi_df.head())
# print(len(fungi_df))
# fungi_df_1024 = fungi_df.loc[fungi_df['length of sequence']<=1024]
#
# print(fungi_df_1024.head())

# #  print(pd.Series(final_df["label"]).value_counts().reset_index())
#
# sample_df = pd.concat([normal_random_df, toxin_random_df, virulence_df])
# # # # print(sample_df)
# # # # print(pd.Series(sample_df["label"]).value_counts().reset_index())
# # # #
# # # sample_df.to_csv("HMP_1024_4class_4255_instance.csv")
# # # # final_df.to_csv("HMP_balanced_0912.csv")
# # #
# sample_df.to_csv("fungi_1024_3class_4609_instance.csv")
# #
# #
# #
# # normal_random_df = normal_df.sample(100)
# # resistance_random_df = resistance_df.sample(100)
# # secretion_random_df = secretion_df.sample(100)
# # toxin_random_df = toxin_df.sample(100)
#
# #

#
# # # use only when needed to create dataset
# df = pd.read_csv("datasets/fungi_4609/fungi_1024_3class_4609_instance.csv")
# print(pd.Series(df["label"]).value_counts().reset_index())
# dataset_split(df, "datasets/fungi_4609/")

# df_new = df.iloc[:,[5,6,7,8,9,10]]
# print(df_new)
# df_new = pd.read_csv("datasets/fungi_32_species/original_data/fungi_1024_3class_32organism_test.csv")
# index_no = []
# for i in range(len(df_new["id"])):
#    index_no.append(i)
# df_new["tensor id"] = index_no
# df_new.to_csv("datasets/fungi_32_species/original_data/fungi_1024_3class_32organism_inference.csv", index=False)

# df = give_index(df)
# df = df_new[df_new["label"] != 0]
# df.to_csv("datasets/fungi_32_species/original_data/fungi_1024_3class_32organism_no_normal_class_inference.csv", index=False)

### BELOW CODE IS FOR MERGING INFERENCE RESULTS WITH THEIR INFORMATION
# df = pd.read_csv("datasets/bacteria_inference/Salmonella_typhimurium_(strainLT2).csv")
# df_compare = pd.read_csv("results/Salmonella_typhimurium_(strainLT2)_1225.csv")
# print(df_compare)
# df_compare = df_compare.drop('Unnamed: 0', axis=1)

# df_new = tensor_id_merge(df, df_compare)
# print(df)

# print(df_new)
# df_new.to_csv("results/Salmonella_typhimurium_(strainLT2)_1225_results.csv", index=False)
# # df_new["label"] = df_new["label"].replace(0, "normal")
# # df_new["label"] = df_new["label"].replace(1, "toxin")
# # df_new["label"] = df_new["label"].replace(2, "virulence")
# # df_new.to_csv("results/fungi_32_species/balanced_4606_peft/fungi_1024_3class_32organism_inference_results_ALL.csv", index=False)
#

####################3

# df = pd.read_csv("results/fungi_32_species/balanced_4606_peft/fungi_1024_3class_32organism_inference_results_ALL.csv")
# print(len(df))
# # df_normal = df[df["label"] == "normal"]
# # print(len(df_normal))

# df_new = df[df["label"] != "normal"]
# print(df_new)
# df_possible_normal = df_new[df["normal"] > 0.5]

# df_possible_normal.to_csv("results/fungi_32_species/balanced_4606_peft/"
#                          "fungi_1024_3class_32organism_no_normal_inference_results_possible_normal.csv", index=False)

# print(len(df_possible_normal))
# df_possible_toxin = df_normal[df["toxin"] > 0.5]
# df_possible_virulence = df_normal[df["virulence"] > 0.5]

# print(df_possible_toxin)
# df_possible_toxin.to_csv("results/fungi_32_species/balanced_4606_peft/"
#                          "fungi_1024_3class_32organism_inference_results_possible_toxin.csv", index=False)
# print(df_possible_virulence)
# df_possible_virulence.to_csv("results/fungi_32_species/balanced_4606_peft/"
#                              "fungi_1024_3class_32organism_inference_results_possible_virulence.csv", index=False)





# df_new = df_new.iloc[:,[2,3,4,5,6,7,8,10,11,12]]
# print(df_new)
# df2["label"] = df2["label"].replace(1, "toxin")
# df2["label"] = df2["label"].replace("2", "virulence")
# print(df2[df2["label"]=="virulence"])
# df_new = df2.iloc[:,[3,4,5,6,7,8,9,10,11,12]]
# df2.to_csv("datasets/fungi_4609/fungi_1024_3class_30organisms_test_no_normal_class_inference_results.csv", index=False)

# print(df2)
# print(df2.sort_values("toxin").head(10))
# print(df2.sort_values("virulence").head(10))
#
# toxin_top_10 = df2.sort_values("toxin").head(10)
# virulence_top_10 = df2.sort_values("virulence").head(10)
#
# top_10 = pd.concat([toxin_top_10, virulence_top_10])
#
#
# top_10 = top_10.iloc[:,[4,5,6,7,8,9,10,11,12,13]]
#
# print(top_10)
#
# top_10.to_csv("results/fungi_4609/top10_inference.csv", index=False)


def give_index(df_new):
    # df_new = df.iloc[:, [2, 3, 4, 5, 6, 10]]    print(df_new)
    index_no = []
    for i in range(len(df_new["id"])):
        index_no.append(i)
    df_new["tensor id"] = index_no

    return df_new


# def exempt_organism(df, organism_list):
#     for item in df["description"]:
#         for organism_name in organism_list:
#             if organism_name in item:
#                 print(organism_name)
#                 print(item)
#
#                 break


#
# organism_list = [
# "Alternaria alternata", "Annulohypoxylon truncatum", "Arthrinium camelliae-sinensis",
#     "Aspergillus niger", "Bjerkandera adusta", "Chaetomium globosum", "Cladosporium cladosporioides",
#     "Coprinellus radians ", "Fusarium equiseti", "Fusarium fujikuroi", "Fusarium incarnatum",
#     "Fusarium proliferatum", "Hypocrea lutea", "Irpex lacteus", "Microdochium albescens",
#     "Moesziomyces aphidis", "Neurospora tetrasperma", "Penicillium brasilianum", "Penicillium chrysogenum",
#     "Penicillium glabrum", "Penicillium oxalicum", "Penicillium rolfsii", "Penicillium sclerotiorum",
#     "Phanerochaete sordida", "Phlebia tremellosa", "Rhodotorula dairenensis", "Rhodotorula glutinis",
#     "Schizophyllum commune", "Trametes versicolor", "Trichoderma harzianum"
# ]
# organism_list = [
#     "Arthrinium camelliae-sinensis", "Cladosporium cladosporioides",
#     "Moesziomyces aphidis", "Rhodotorula dairenensis"]
#
# #
# print(len(organism_list))
# df = pd.read_csv("fungi_1024_3class_30organisms_test.csv")
# print(df)
# #
# # # exempt_organism(df, organism_list)
# organism_list = [
#
# ]
# def remove_duplicates(df, organism_list):
#     test_df = pd.DataFrame(columns=['id', 'description', 'sequence', 'length of sequence'])
#     for organism_name in organism_list:
#         matching_rows = df[df['description'].str.contains(organism_name, case=False, na=False)]
#         test_df = pd.concat([test_df, matching_rows])
#
#     train_df = pd.concat([df, test_df])
#
#     train_df = train_df.drop_duplicates(keep=False)
#
#     return train_df

# df = pd.read_csv("datasets/HMP/HMP_1024_ALL.csv")
# test_df= pd.DataFrame(columns=['bacteria species', 'id', 'description', 'sequence', 'length of sequence'])
#
# for organism_name in organism_list:
#     matching_rows = df[df['description'].str.contains(organism_name, case=False, na=False)]
#     test_df = pd.concat([test_df, matching_rows])
#
#
# train_df = pd.concat([df, test_df])
#
# train_df = train_df.drop_duplicates(keep=False)

# test_df.to_csv("fungi_1024_3class_4organisms_test.csv")
# train_df.to_csv("fungi_1024_3class_30classes_train.csv")


# set tensor id as index and merge 2 df



def make_output_csv(df):
    df2 = pd.read_excel("datasets/fungi/30_fungi_list.xlsx")
    fungi_name_list = df2["학명"].tolist()
    print(fungi_name_list)

    save_csv = pd.DataFrame(columns=['fungi species', 'normal', 'toxin', 'virulence'])
    # print(save_csv)
    for fungi_name in fungi_name_list:
        contain_values = df[df["fungi species"].str.contains(fungi_name[:-1])]
        label_counts = contain_values["label"].value_counts()

        label_counts = label_counts.reset_index().transpose()
        # contain_values.columns = ['normal', 'toxin', 'virulence']
        label_counts = label_counts.rename(columns=label_counts.iloc[0]).iloc[1:, :]
        label_counts.insert(0, "fungi species", fungi_name)
        save_csv = pd.concat([save_csv, label_counts])

    save_csv.to_csv("results/fungi_4609/30_species_labels.csv", index=False)
    print(save_csv)


def change_label_string(df1):
    df1["label"] = df1["label"].replace(0, "normal")
    df1["label"] = df1["label"].replace(1, "toxin")
    df1["label"] = df1["label"].replace(2, "virulence")

    return df1


# df = pd.read_csv("datasets/fungi_uniprot_all_231012.csv")
# # df.to_csv("datasets/fungi_uniprot_all_231012.csv", index=False)
# print(df.head())
# print(len(df))
# df = pd.read_csv("datasets/fungi_4609/fungi_1024_3class_30classes_train.csv")
# df1 = pd.read_csv("datasets/fungi_4609/fungi_1024_3class_30organisms_test.csv")
# df2 = pd.read_csv("datasets/fungi_4609/fungi_1024_3class_30organisms_test_no_normal_class_inference_results.csv")

# df_new = df.iloc[:,[3,4,5,6,7]]
# print(df_new.head())
#
# print(len(df_new))
# fungi_df_1024 = df_new.loc[df_new['length of sequence']<=1024]
# print(len(fungi_df_1024))
# label_list = []
# for item in fungi_df_1024["description"]:
#     if "virul" in item:
#         # print("VIRULENCE", item)
#         label_list.append(2)
#     elif 'toxi' in item:
#         # print("TOXIN", item)
#         if 'toxin-antitoxin' in item:
#             label_list.append(0)
#         else:
#             label_list.append(1)
#     else:
#         label_list.append(0)
#
# fungi_df_1024["label"] = label_list
# fungi_df_1024.rename(columns = {'fungi species' : 'fungi species and strand'}, inplace = True)
# print(fungi_df_1024)
# fungi_species_list = []
# for fungi_species in fungi_df_1024["fungi species and strand"]:
#     split_string = str(fungi_species).split(" ")
#     if len(split_string) == 1:
#         fungi_species_list.append("nan")
#     else:
#         combined_string = split_string[0] + " " + split_string[1]
#         fungi_species_list.append(combined_string)
#
# fungi_df_1024["fungi species"] = fungi_species_list
# print("before removing proteins with only family")
# print(len(fungi_df_1024))
# fungi_df_1024 = fungi_df_1024[fungi_df_1024["fungi species"] != "nan"]
# print("after removing proteins with only family")
# print(len(fungi_df_1024))
# check_labels_count(fungi_df_1024)
# index_no = []
# for i in range(len(fungi_df_1024["id"])):
#    index_no.append(i)
# fungi_df_1024["tensor id"] = index_no
# fungi_df_1024.to_csv("datasets/fungi_uniprot_all_231012.csv")
# print(fungi_df_1024.head())

# print(df)

# df2 = pd.read_excel("datasets/fungi/30_fungi_list.xlsx")
# fungi_name_list = df2["학명"].tolist()
# print(fungi_name_list)
# print(len(fungi_name_list))
#
# test_df = pd.DataFrame(columns=['id', 'description', 'sequence', 'length of sequence', 'fungi species and strand',
#                                 'label', 'fungi species'])
#
# for fungi_name in fungi_name_list:
#     contain_values = df[df["fungi species"].str.contains(fungi_name[:-1])]
#     test_df = pd.concat([test_df, contain_values])
#
# print(test_df)
# train_df = pd.concat([df, test_df])
#
# train_df = train_df.drop_duplicates(keep=False)
#
# print(train_df)
# print(len(train_df))
# print(len(test_df))
# # print(test_df["fungi species"].value_counts())
# # #
# test_df.to_csv("fungi_1024_3class_30organisms_test.csv", index=False)
# train_df.to_csv("fungi_1024_3class_30organisms_train.csv", index=False)

# df = pd.read_csv("datasets/fungi_30_species/original_data/fungi_1024_3class_30organisms_test.csv")
# print(df.head())
# df_new = df.iloc[:,[23,4,5,6,7,8,]]
# check_labels_count(df)
# index_no = []
# for i in range(len(df["id"])):
#    index_no.append(i)
# df["tensor id"] = index_no
# df.to_csv("datasets/fungi_30_species/original_data/fungi_1024_3class_30organisms_test.csv", index=False)


# df = pd.read_csv("fungi_1024_3class_30organisms_train.csv")
# print(len(pd.unique(df['fungi species'])))
# print(len(pd.unique(df["fungi species and strand"])))
# check_labels_count(df)
# index_no = []
# #
# for i in range(len(df["id"])):
#    index_no.append(i)
# df["tensor id"] = index_no
# print(df)
# df.to_csv("fungi_1024_3class_30organisms_train.csv", index=False)
#
# df_by_labels = df.groupby(df.label)
# virulence_df = df_by_labels.get_group(2)
# toxin_df = df_by_labels.get_group(1)
# normal_df = df_by_labels.get_group(0)
#
# virulence_df = virulence_df.sample(4609)
# toxin_random_df = toxin_df.sample(4609)
# normal_random_df = normal_df.sample(4609)
#
# sample_df = pd.concat([normal_random_df, toxin_random_df, virulence_df])
#
# sample_df.to_csv("datasets/fungi_30_species/original_data/fungi_1024_3class_30organisms_train_balanced_4609.csv", index=False)


# df = pd.read_csv("datasets/fungi_30_species/original_data/fungi_1024_3class_30organisms_train_balanced_4609.csv")
# check_labels_count(df)
# dataset_split(df, "datasets/fungi_30_species/balanced_4609/")

# df = pd.read_csv("datasets/fungi_30_species/original_data/fungi_1024_3class_30organisms_train.csv")
# train_df = pd.read_csv("datasets/fungi_30_species/balanced_4609/train.csv")
#
# test_df = pd.read_csv("datasets/fungi_30_species/balanced_4609/test.csv")
# valid_df = pd.read_csv("datasets/fungi_30_species/balanced_4609/valid.csv")
# df = pd.read_csv("datasets/fungi_30_species/original_data/fungi_1024_3class_30organisms_train.csv")

def find_organism(df):
    df2 = pd.read_excel("datasets/fungi/30_fungi_list.xlsx")
    fungi_30_list = df2["학명"].tolist()
    fungi_name_list = ["Aspergilus fumigatus ", "Penicillium sp. ", "Cladosporium cladosporioides "]
    fungi_name_list = fungi_30_list+fungi_name_list
    print(fungi_name_list)

    # save_csv = pd.DataFrame(columns=['fungi species', '0', '1', '2'])
    save_csv = pd.DataFrame(columns=['fungi species', 'normal', 'toxin', 'virulence'])
    # print(save_csv)
    for fungi_name in fungi_name_list:
        # contain_values = df[df["fungi species"].str.contains(fungi_name)]
        contain_values = df[df["fungi species"].str.contains(fungi_name[:-1])]
        label_counts = contain_values["label"].value_counts()

        label_counts = label_counts.reset_index().transpose()
        # contain_values.columns = ['normal', 'toxin', 'virulence']
        label_counts = label_counts.rename(columns=label_counts.iloc[0]).iloc[1:, :]
        label_counts.insert(0, "fungi species", fungi_name)
        save_csv = pd.concat([save_csv, label_counts])

    # save_csv.to_csv("datasets/fungi_32_species/original_data/32종 데이터.csv")
    print(save_csv)

# df = pd.read_csv("datasets/fungi_uniprot_all_231012.csv")
# print(len(df))
# print("length of fungi dataset without 1024")
#
# test_df = pd.DataFrame(columns=['id', 'description', 'sequence', 'length of sequence', 'fungi species and strand',
#                                 'label', 'fungi species'])
#
# df2 = pd.read_excel("datasets/fungi/30_fungi_list.xlsx")
# fungi_30_list = df2["학명"].tolist()
# fungi_name_list = ["Aspergilus fumigatus ", "Penicillium sp. "]
# fungi_name_list = fungi_30_list + fungi_name_list
# print(fungi_name_list)
# for fungi_name in fungi_name_list:
#     contain_values = df[df["fungi species"].str.contains(fungi_name[:-1])]
#     test_df = pd.concat([test_df, contain_values])
#
# train_df = pd.concat([df, test_df])
#
# train_df = train_df.drop_duplicates(keep=False)
#
# print(len(train_df))
# print("length of trainable df")
# print(len(test_df))
# print("length of test df - 33 species")
# train_df["label"] = train_df["label"].astype("int64")
# test_df["label"] = test_df["label"].astype("int64")
# # find_organism(test_df)
#
# check_labels_count(train_df)
# check_labels_count(test_df)
# train_df.to_csv("datasets/fungi_32_species/original_data/fungi_1024_3class_32organism_train.csv", index=False)
# test_df.to_csv("datasets/fungi_32_species/original_data/fungi_1024_3class_32organism_test.csv", index=False)
#




# find_organism(df)
# df = pd.read_csv("datasets/fungi_uniprot_all_231012.csv")
# print(len(df))
# check_labels_count(test_df)
# check_labels_count(valid_df)

# print(df)
#
# # df = pd.concat([df, test_df])
# # df = pd.concat([df,train_df])
# # df = df.drop_duplicates(subset=['tensor id'])
#
# test_list = test_df["tensor id"].tolist()
# vald_list = valid_df["tensor id"].tolist()
# tensor_id_list = test_list+vald_list
# print(tensor_id_list)
#
# test_df = pd.DataFrame(columns=['id', 'description', 'sequence', 'length of sequence', 'fungi species and strand',
#                                 'label', 'fungi species'])
#
# for fungi_name in tensor_id_list:
#     contain_values = df[df["tensor id"] == fungi_name]
#     test_df = pd.concat([test_df, contain_values])
#
# print(test_df)
# train_df = pd.concat([df, test_df])
#
# train_df = train_df.drop_duplicates(keep=False)
# print(train_df)
# train_df['tensor id'] = train_df['tensor id'].astype('int64')
# train_save = train_df[["sequence", "label", "tensor id"]]
# check_labels_count(train_save)
# train_save.to_csv("datasets/fungi_30_species/imbalanced/train.csv", index=False)

# train_df = pd.read_csv("datasets/fungi_32_species/original_data/fungi_1024_3class_32organism_train.csv")
# train_df = give_index(train_df)
# dataset_split(train_df, "datasets/fungi_32_species/imbalanced/")

# test_df = pd.read_csv("datasets/fungi_32_species/original_data/fungi_1024_3class_32organism_test.csv")
# test_df = change_label_string(test_df)
# check_labels_count(test_df)
# find_organism(test_df)

# df = pd.read_csv("datasets/fungi_32_species/original_data/fungi_1024_3class_32organism_train.csv")
# df = give_index(df)
# print(df)
# df_by_labels = df.groupby(df.label)
# virulence_df = df_by_labels.get_group(2)
# toxin_df = df_by_labels.get_group(1)
# normal_df = df_by_labels.get_group(0)
#
# virulence_df = virulence_df.sample(4606)
# toxin_random_df = toxin_df.sample(4606)
# normal_random_df = normal_df.sample(4606)
# sample_df = pd.concat([normal_random_df, toxin_random_df, virulence_df])
# sample_df.to_csv("datasets/fungi_32_species/balanced_4606/fungi_1024_3class_32organism_4606_balanced.csv", index=False)
# dataset_split(sample_df,"datasets/fungi_32_species/balanced_4606/")

# df = pd.read_csv("datasets/fungi_32_species/balanced_4606/test.csv")
#
#

# df2 = pd.read_csv("results/fungi_32_species/balanced_4606_peft/fungi_1024_3class_32organism_inference_results_ALL.csv")
# print(df2.sort_values("toxin").head(10))
# print(df2.sort_values("virulence").head(10))
# print(df2.sort_values('normal').head(10))
# #
# toxin_top_10 = df2.sort_values("toxin").head(10)
# virulence_top_10 = df2.sort_values("virulence").head(10)
# normal_top10 = df2.sort_values("normal").head(10)
# #
# top_10 = pd.concat([normal_top10, toxin_top_10, virulence_top_10])
#
#
# top_10 = top_10.iloc[:,[4,5,6,7,8,9,10,11,12,13]]
#
# print(top_10)
#
# top_10.to_csv("results/fungi_32_species/balanced_4606_peft/"
#               "fungi_1024_3class_32organism_inference_results_ALL_top10.csv", index=False)


def get_top10(df):
    # print(df.sort_values("toxin", ascending=False).head(10))
    # print(df.sort_values("virulence", ascending=False).head(10))
    print(df.sort_values('normal', ascending=False).head(10))
    # toxin_top_10 = df.sort_values("toxin", ascending=False).head(10)
    # virulence_top_10 = df.sort_values("virulence", ascending=False).head(10)
    normal_top10 = df.sort_values("normal", ascending=False).head(10)
    # top_10 = pd.concat([normal_top10, toxin_top_10, virulence_top_10])

    return normal_top10
# root_path = "results/fungi_32_species/balanced_4606_peft/"
# df1 = pd.read_csv(root_path+"fungi_1024_3class_32organism_inference_results_possible_toxin.csv")
# df2 = pd.read_csv(root_path+"fungi_1024_3class_32organism_inference_results_possible_virulence.csv")
# df3 = pd.read_csv(root_path+"fungi_1024_3class_32organism_no_normal_inference_results_possible_normal.csv")
#
#
# # toxin_top = get_top10(df1)
# # virul_top = get_top10(df2)
# normal_top = get_top10(df3)
#
# # toxin_top.to_csv(root_path+"fungi_1024_3class_32organism_inference_results_possible_toxin_top10.csv", index=False)
# # virul_top.to_csv(root_path+"fungi_1024_3class_32organism_inference_results_possible_virulence_top10.csv", index=False)
# normal_top.to_csv(root_path+"fungi_1024_3class_32organism_no_normal_inference_results_possible_normal_top10.csv", index=False)

#
# df = pd.read_csv("results/fungi_32_species/balanced_4606_peft/fungi_1024_3class_32organism_inference_results_ALL.csv")
# print(df)


# find_organism(df)

# fungi_name_list = ["Alternaria alternata", "Bjerkandera adusta", "Cladosporium cladosporioides",
#                    "Fusarium proliferatum", "Penicillium brasilianum"]
# print(fungi_name_list)
#
# # save_csv = pd.DataFrame(columns=['fungi species', '0', '1', '2'])
# save_csv = pd.DataFrame(columns=['fungi species', 'normal', 'toxin', 'virulence'])
# pred_csv = pd.DataFrame(columns=['fungi species and strand','description', 'normal', 'toxin', 'virulence', 'label'])
# # print(save_csv)
# for fungi_name in fungi_name_list:
#     # contain_values = df[df["fungi species"].str.contains(fungi_name)]
#     contain_values = df[df["fungi species"].str.contains(fungi_name)]
#     pred_csv = pd.concat([pred_csv, contain_values])
#     # label_counts = contain_values["label"].value_counts()
#     #
#     # label_counts = label_counts.reset_index().transpose()
#     # # contain_values.columns = ['normal', 'toxin', 'virulence']
#     # label_counts = label_counts.rename(columns=label_counts.iloc[0]).iloc[1:, :]
#     # label_counts.insert(0, "fungi species", fungi_name)
#     # save_csv = pd.concat([save_csv, label_counts])
#
#
# # print(save_csv)
# pred_csv = pred_csv.iloc[:,[0,1,2,3,4,5]]
# print(pred_csv)
# pred_csv.to_csv("results/fungi_32_species/balanced_4606_peft/5species_results/original4.csv", index=False)

# def round_values(df):


# df = pd.read_csv("results/fungi_32_species/balanced_4606_peft/5species_results/original4.csv")
#
# print(df)
# toxicdf = df[df["label"] == "normal"]
# normaldf = df[df["label"] != "normal"]
#
# possible_toxin = toxicdf.sort_values('toxin', ascending=False).round(4)
# possible_toxin.to_csv("results/fungi_32_species/balanced_4606_peft/5species_results/possible_toxin.csv", index=False)
# print(len(possible_toxin))
#
# possible_virul = toxicdf.sort_values('virulence', ascending=False).round(4)
# possible_virul.to_csv("results/fungi_32_species/balanced_4606_peft/5species_results/possible_virulence.csv", index=False)
# print(len(possible_virul))
#
#
# possible_normal = normaldf.sort_values('normal', ascending=False).round(4)
# possible_normal.to_csv("results/fungi_32_species/balanced_4606_peft/5species_results/possible_normal.csv", index=False)
# print(len(possible_normal))

