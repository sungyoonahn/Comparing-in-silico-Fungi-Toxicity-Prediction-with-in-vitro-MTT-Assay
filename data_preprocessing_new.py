import pandas as pd
from sklearn import preprocessing
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
import os
from sklearn.model_selection import train_test_split
import sys, re
from Bio import SeqIO

pd.set_option('display.max_columns', None)

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

root = "datasets/"
tremble_path = root+"TremBLE(UNIPROT)/Bacteria/uniprot_bacteria_temble.fasta"
swissprot_path = root+"SwissProt(UNIPROT)/uniprot_bacteria_1224.fasta"
HMP_path = root + "HMP/HMP_1024_ALL.csv"
bacteria_save_path = root+"SwissProt(UNIPROT)/"
tremble_group_path = 'datasets/TremBLE(UNIPROT)/Bacteria/split_data'
tremble_save_path = 'datasets/TremBLE(UNIPROT)/Bacteria/'
tester_df_path = root+'tester_df/'

# 0:normal(else) / 1:secretion / 2: resistance / 3: toxin / 4: anti-toxin / 5: kill / 6:phage / 7:virulence / 8: invasion / 9: effector / 10: putative / 11: hypothetical, unknown, uncharacterized
class_list = ["uncha", "unknown", "hypothe", "kill", "inva", "virul", "toxi", "resi", "secre", "phage", "puta", "effector"]

tester_df = pd.read_csv(tester_df_path+'tester_df.csv')

def check_labels_count(df):
    print("checking count for each labels...\n-------")
    print(pd.Series(df["label"]).value_counts().reset_index())
    
    
# def find_matching_words(sentence, class_list):
    # matching_words = []
    # for prefix in class_list:
    #     if prefix in sentence:
    #         print(prefix)
    #         matching_words.append(prefix)
    # for item in matching_words:
    #     print(item)
    #     if "uncha" in item:
    #         label_list.append(11)
    #     elif "unknown" in item:
    #         label_list.append(11)
    #     elif "hypothe" in item:
    #         label_list.append(11)
    #     elif "puta" in item:
    #         label_list.append(10)
    #     elif "effector" in item:
    #         label_list.append(9)
    #     elif "inva" in item:
    #         # print("INVASION", item)
    #         label_list.append(8)
    #     elif "virul" in item:
    #         # print("VIRULENCE", item)
    #         label_list.append(7)
    #     elif "phage" in item:
    #         label_list.append(6)
    #     elif "kill" in item:
    #         # print("KILL", item)
    #         label_list.append(5)

    #     elif 'toxi' in item:
    #         # print("TOXIN", item)
    #         if 'anti' in sentence:
    #             label_list.append(4)
    #         else:
    #             label_list.append(3)

    #     elif 'resi' in item:
    #         # print("RESISTANT", item)
    #         label_list.append(2)
    #     elif 'secre' in item:
    #         # print("SECRETION", item)
    #         label_list.append(1)
    #     else:
    #         label_list.append(0)
    # return matching_words, label_list

def label_data_bacteria(df):
    print("labelling bacteria sequences...\n-------")
    label_list = []
    
    for index, row in df.iterrows():
        for prefix in class_list:
            matching_words = []
            if prefix in row.description:
                matching_words.append(prefix)
                print(matching_words)
                break
            
        # for item in matching_words:      
        #     if "uncha" in item:
        #         label_list.append(11)
        #     elif "unknown" in item:
        #         label_list.append(11)
        #     elif "hypothe" in item:
        #         label_list.append(11)
        #     elif "puta" in item:
        #         label_list.append(10)
        #     elif "effector" in item:
        #         label_list.append(9)
        #     elif "inva" in item:
        #         # print("INVASION", item)
        #         label_list.append(8)
        #     elif "virul" in item:
        #         # print("VIRULENCE", item)
        #         label_list.append(7)
        #     elif "phage" in item:
        #         label_list.append(6)
        #     elif "kill" in item:
        #         # print("KILL", item)
        #         label_list.append(5)
        #     elif 'toxi' in item:
        #         # print("TOXIN", item)
        #         if 'anti' in item:
        #             label_list.append(4)
        #         else:
        #             label_list.append(3)
        #     elif 'resi' in item:
        #         # print("RESISTANT", item)
        #         label_list.append(2)
        #     elif 'secre' in item:
        #         # print("SECRETION", item)
        #         label_list.append(1)
        #     else:
        #         label_list.append(0)
        


    # df["label"] = label_list
    # print("done labelling sequences!\n-------")
    return df

    
tester_df_labeled = label_data_bacteria(tester_df)
print(tester_df_labeled)
# check_labels_count(tester_df_labeled)