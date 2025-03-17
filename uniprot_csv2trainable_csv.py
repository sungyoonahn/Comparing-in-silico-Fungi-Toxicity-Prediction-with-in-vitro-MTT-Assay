import pandas as pd
import os
import numpy as np
import csv
# secreted KW=0964, label = 1
# Antibiotic resistance (KW-0046), label = 2
# Toxin KW-0800, label = 3
# toxin-antitoxin system KW-1277, label = 4
# Virulence KW-0843, label = 5


# ##### FOR BACTERIA INFERENCE #####
# keyword_labels = {
#     "KW-0964": 1,  
#     "KW-0046": 2,  
#     "KW-0800": 3,  
#     "KW-1277": 4,  
#     "KW-0843": 5  
# }


# def check_labels_count(df):
#     print("checking count for each labels...\n-------")
#     print(pd.Series(df["label"]).value_counts().reset_index())

# # bacteria_name ="Roseomonas mucosa"

# bacteria_data_path = "env_project/datasets/bacteria_inference/1126/"

# for bacteria_name in os.listdir(bacteria_data_path):
#         bacteria_name = bacteria_name.split(".")[0]
#         df = pd.read_csv(bacteria_data_path+bacteria_name+".tsv", delimiter="\t")
#         df['sequence'] = df['Sequence'].apply(lambda x: " ".join(x))

#         index_no = []
#         for i in range(len(df["Sequence"])):
#             index_no.append(i)
#         df["tensor id"] = index_no
#         df['label'] = 0

#         preprocessing = []  
#         for _, row in df.iterrows():  
        
#             if pd.notna(row['Keyword ID']):
#                 keywords = row['Keyword ID'].split(";")
                
#                 matched_labels = [keyword_labels[keyword.strip()] for keyword in keywords if keyword.strip() in keyword_labels]
                
#                 if not matched_labels:  
#                     row['label'] = 0
#                     preprocessing.append(row)
#                 elif len(matched_labels) == 1:
#                     row['label'] = matched_labels[0]
#                     preprocessing.append(row)
#                 else:
#                     for label in matched_labels:
#                         new_row = row.copy() 
#                         new_row['label'] = label 
#                         preprocessing.append(new_row)
#             else:
#                 row['label'] = 0
#                 preprocessing.append(row)

#         processed_df = pd.DataFrame(preprocessing)

#         print(processed_df.head())
#         check_labels_count(processed_df)
#         processed_df.to_csv(bacteria_data_path+bacteria_name+".csv", index=False) 



# ##### FOR FUNGI INFERENCE #####
GO_labels = {
    "GO:0007155": 1,
    "GO:0010838": 1,
    "GO:0006950": 2,
    "GO:0006970": 2,
    "GO:0015893": 3,
    "GO:0042594": 3,
    "GO:0046677": 4,
    "GO:0006281": 4,
}


def check_labels_count(df):
    print("checking count for each labels...\n-------")
    print(pd.Series(df["label"]).value_counts().reset_index())



fungi_data_path = "env_project/datasets/fungi_inference/1126/"
for fungi_name in os.listdir(fungi_data_path):

    fungi_name = fungi_name.split(".")[0]
    df = pd.read_csv(fungi_data_path+fungi_name+".tsv", delimiter="\t")

    df['sequence'] = df['Sequence'].apply(lambda x: " ".join(x))

    index_no = []
    for i in range(len(df["Sequence"])):
        index_no.append(i)
    df["tensor id"] = index_no
    df['label'] = 0
    
    print(df["Gene Ontology IDs"].head())
    preprocessing = []  
    for _, row in df.iterrows():  
        if pd.notna(row['Gene Ontology IDs']):
            keywords = row['Gene Ontology IDs'].split(";")
            
            matched_labels = [GO_labels[keyword.strip()] for keyword in keywords if keyword.strip() in GO_labels]
            
            if not matched_labels:  
                row['label'] = 0
                preprocessing.append(row)
            elif len(matched_labels) == 1:
                row['label'] = matched_labels[0]
                preprocessing.append(row)
            else:
                for label in matched_labels:
                    new_row = row.copy() 
                    new_row['label'] = label 
                    preprocessing.append(new_row)
        else:
            row['label'] = 0
            preprocessing.append(row)

    processed_df = pd.DataFrame(preprocessing)

    print(processed_df.head())
    check_labels_count(processed_df)
    processed_df.to_csv(fungi_data_path+fungi_name+".csv", index=False) 
