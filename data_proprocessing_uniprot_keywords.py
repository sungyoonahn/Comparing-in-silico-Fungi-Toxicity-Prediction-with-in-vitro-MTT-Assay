import pandas as pd
from sklearn import preprocessing
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
import os
from sklearn.model_selection import train_test_split
import sys, re
from Bio import SeqIO
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


root = "datasets/"
tremble_path = root+"TremBLE(UNIPROT)/Bacteria/uniprot_bacteria_temble.fasta"
swissprot_path = root+"SwissProt(UNIPROT)/uniprot_bacteria_1224.fasta"
HMP_path = root + "HMP/HMP_1024_ALL.csv"
bacteria_save_path = root+"SwissProt(UNIPROT)/"
tremble_group_path = 'datasets/TremBLE(UNIPROT)/Bacteria/split_data'
tremble_save_path = 'datasets/TremBLE(UNIPROT)/Bacteria/'
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


def fasta_2_csv_uniprot_keyword(fasta_paths, save_path):
    # This is the code for reading all the fasta files from uniprot, downloaded based on the keyword, can be used for GO as well
    print("reading fasta file...\n-------")
    df = pd.DataFrame()
    for fasta_path in fasta_paths:
        fasta_data=SeqIO.parse(open(fasta_path), "fasta")
        species_strand_list = []
        id_list = []
        seq_list = []
        des_list = []
        seq_len_list = []
        label_list = []
        data_type_list = []
        
        label = fasta_path.split("_")[7]
        data_type = fasta_path.split("_")[8]
        
        for fasta in fasta_data:
            id_list.append(fasta.id)
            seq_list.append(str(fasta.seq))
            des_list.append(fasta.description)
            seq_len_list.append(len(fasta.seq))
            match = re.search(r'OS=(.*?) OX=', fasta.description)
            species_strand_list.append(match.group(1))
            label_list.append(label)
            data_type_list.append(data_type)
    
    
        df_class = pd.DataFrame([x for x in zip(species_strand_list,
                                                id_list, des_list, seq_list, seq_len_list, label_list, data_type_list)],
                                columns=['species and strand', 'id', 'description',
                                        'sequence', 'length of sequence', 'label', 'data type'])
        df = pd.concat([df, df_class])
    print("read fasta file and now saving to df...\n-------")
    
    df.to_csv(save_path+"TREMBL_1008_5_class.csv", index=False)
    
    print("saved fasta to df format!\n-------")
    return df


# drops sequences with only family info
def sequences_family_drop(df):
    print("dropping sequences with only family...\n-------")
    species_list = []
    for species in df["species and strand"]:
        split_string = str(species).split(" ")
        if len(split_string) == 1:
            species_list.append("nan")
        else:
            combined_string = split_string[0] + " " + split_string[1]
            species_list.append(combined_string)
    
    
    df["species"] = species_list
    df1 = df[df["species"] != "nan"]
    print("Number of sequences with only family: ",(len(df)-len(df1)))
    print("done dropping sequences with only family!\n-------")
    
    return df1

# checks for the total count of each label
def check_labels_count(df):
    print("checking count for each labels...\n-------")
    print(pd.Series(df["label"]).value_counts().reset_index())
    
#drops rows longer than 1022
def drop_length(df):
    print("removing rows longer than 1022...\n-------")
    df1 = df[df["length of sequence"].astype(int)<=1022]
    print("removed rows longer than 1022!\n-------")
    return df1

# checks for the total count of each label and plot the distribution
# def plot_check_labels_count(df):
#     print("checking count for each labels...\n-------")
#     df.groupby([df.label, 'label']).count().plot(kind='bar')
#     plt.show()

def drop_inference_data(df, organism_list):
    print("removing rows belonging to the 71 species...\n-------")
    
    test_df = pd.DataFrame(columns=['species and strand', 'id', 'description', 'sequence', 'length of sequence', 'label', 'species'])
    for organism_name in organism_list:
        matching_rows = df[df['description'].str.contains(organism_name, case=False, na=False)]
        test_df = pd.concat([test_df, matching_rows])

    train_df = pd.concat([df, test_df])

    train_df = train_df.drop_duplicates(keep=False)
    print("length of trainable data: ", len(train_df))
    print("length of inference data: ", len(test_df))
    print("removed rows belonging to the 71 species!\n-------")

    return train_df, test_df



# splits df to train, validation and test sets
def dataset_split(df, save_dir):
    train_set, test_set = train_test_split(df, test_size=0.3, random_state=42, stratify=
    df["label"].values.tolist())
    val_set, test_set = train_test_split(test_set, test_size=0.333, random_state=42, stratify=
    test_set["label"].values.tolist())
    my_train = train_set[["sequence", "label", "tensor id"]]
    my_valid = val_set[["sequence", "label", "tensor id"]]
    my_test = test_set[["sequence", "label", "tensor id"]]

    my_train.to_csv(save_dir + "train.csv", index = False)
    check_labels_count(my_train)
    my_valid.to_csv(save_dir + "valid.csv", index = False)
    check_labels_count(my_valid)
    my_test.to_csv(save_dir + "test.csv", index = False)
    check_labels_count(my_test)

def merge_inferenece(df, df_compare):
### BELOW CODE IS FOR MERGING INFERENCE RESULTS WITH THEIR INFORMATION

    print(df_compare)
    df_compare = df_compare.drop('Unnamed: 0', axis=1)

    df_new = tensor_id_merge(df, df_compare)
    print(df)

    print(df_new)
    return df_new
    

def tensor_id_merge(df1, df2):
    df3 = pd.merge(df1, df2, on='tensor id', how='inner')

    return df3


def batch_iterator(iterator, batch_size):
    entry = True  # Make sure we loop once

    while entry:
        batch = []

        while len(batch) < batch_size:
            try:
                entry = next(iterator)
            except StopIteration:
                entry = False

            if not entry:
                # End of file
                break

            batch.append(entry)

        if batch:
            yield batch
            

def split_fast2_batch():
    fasta_data= SeqIO.parse(tremble_path, "fasta")
    tremble_group_path = 'datasets/TremBLE(UNIPROT)/Bacteria/split_data'
        
    for i, batch in enumerate(batch_iterator(fasta_data, 10000), start=1):
        filename = 'datasets/TremBLE(UNIPROT)/Bacteria/split_data/group_{}.fasta'.format(i)
        count = SeqIO.write(batch, filename, 'fasta')
        print('Wrote {} records to {}'.format(count, filename))
        
        
# 0:normal(else) / 1:secretion / 2: resistance / 3: toxin / 4: anti-toxin / 5:virulence / 6: kill / 7: invasion / 8:phage / 9: effector / 10: putative / 11: hypothetical, unknown, uncharacterized

def split_data(df):
    df_by_labels = df.groupby(df.label)
    normal_df = df_by_labels.get_group(0)
    secretion_df = df_by_labels.get_group(1)
    resistant_df = df_by_labels.get_group(2)
    toxin_df = df_by_labels.get_group(3)
    anti_toxin_df = df_by_labels.get_group(4)
    virulence_df= df_by_labels.get_group(5)
    # kill_df = df_by_labels.get_group(6)
    # invasion_df = df_by_labels.get_group(7)
    # phage_df = df_by_labels.get_group(8)
    # effector_df = df_by_labels.get_group(9)
    # putative_df = df_by_labels.get_group(10)
    # unknown_df = df_by_labels.get_group(11)
    
    
    normal_random_df = normal_df.sample(1754)
    # secretion_random_df = secretion_df.sample(100)
    # resistant_random_df = resistant_df.sample(100)
    # toxin_random_df = toxin_df.sample(100)
    # anti_toxin_random_df = anti_toxin_df.sample(100)
    # virulence_random_df = virulence_df.sample(100)
    # putatuve_random_df = putative_df.sample(14006)
    # phage_random_df = phage_df.sample(14006)
    # unknow_random_df = unknown_df.sample(14006)
    # # sample_df = pd.concat([normal_random_df, secretion_random_df, resistant_random_df, toxin_random_df, virulence_df, invasion_df, 
    #                        kill_df, phage_random_df,anti_toxin_random_df])
    
    sample_df = pd.concat([normal_random_df, secretion_df, resistant_df, toxin_df, anti_toxin_df, virulence_df])
    return sample_df
    

def batch_training_data(df):
    df_by_labels = df.groupby(df.label)
    # normal_df = df_by_labels.get_group(0)
    secretion_df = df_by_labels.get_group(1)
    resistant_df = df_by_labels.get_group(2)
    toxin_df = df_by_labels.get_group(3)
    anti_toxin_df = df_by_labels.get_group(4)
    virulence_df= df_by_labels.get_group(5)
    invasion_df = df_by_labels.get_group(6)
    kill_df = df_by_labels.get_group(7)
    phage_df = df_by_labels.get_group(8)
    putative_df = df_by_labels.get_group(9)
    unknown_df = df_by_labels.get_group(10)
    
    data_size = len(df)
    putative_data_size = len(putative_df)
    unknown_data_size = len(unknown_df)
    batchable_training_data_size = data_size - putative_data_size - unknown_data_size
    num_class = 8
    data_mean = int(batchable_training_data_size/num_class)
    print(data_mean)
    for i in range(num_class):
        # normal_random_df = normal_df.sample(data_mean)
        secretion_random_df = secretion_df.sample(int(len(secretion_df)/num_class))
        resistant_random_df = resistant_df.sample(int(len(resistant_df)/num_class))
        toxin_random_df = toxin_df.sample(int(len(toxin_df)/num_class))
        anti_toxin_random_df = anti_toxin_df.sample(int(len(anti_toxin_df)/num_class))
        virulence_random_df= virulence_df.sample(int(len(virulence_df)/num_class))
        invasion_random_df = invasion_df.sample(int(len(invasion_df)/num_class))
        kill_random_df = kill_df.sample(int(len(kill_df)/num_class))
        phage_random_df = phage_df.sample(int(len(phage_df)/num_class))
        # putatuve_random_df = putative_df.sample(48205)
        # unknow_random_df = unknown_df.sample(48205)
        sample_df = pd.concat([secretion_random_df, resistant_random_df, toxin_random_df, virulence_random_df, invasion_random_df, 
                           kill_random_df, phage_random_df,anti_toxin_random_df])
        sample_df.to_csv(tremble_save_path+"batch_data/batch_"+str(i)+".csv", index=False)
        

# checks for the total count of each label and plot the distribution
def plot_check_labels_count(df):
    print("checking count for each labels...\n-------")
    df.groupby([df.label, 'label']).count().plot(kind='bar')
    plt.show()

def merge_Swissprot_Keyword(df1, df2):
    # function to get class 0 normal, merge swissprot df with keyword df and extract df without keywords
    df2_filtered = df2[~df2["sequence"].isin(df1["sequence"])]

    return df2_filtered

def give_tensor_id(df):
    index_no = []
    for i in range(len(df["sequence"])):
       index_no.append(i)
    df["tensor id"] = index_no
    
    return df
def plot_check_labels_count(df):
    print("checking count for each labels...\n-------")
    # Calculate the value counts for the "label" column
    label_counts = df["label"].value_counts()

    # Define colors for each class
    color_mapping = {
        "secretion": "skyblue",
        "resistance": "salmon",
        "toxin": "lightgreen",
        "anti-toxin": "gold",
        "virulence": "orchid"
    }
    colors = [color_mapping[label] for label in label_counts.index]
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Bar Plot with different colors
    label_counts.plot(kind="bar", color=colors, ax=axes[0])
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=0)  # Rotate x-axis labels

    # Pie Chart with different colors
    label_counts.plot(kind="pie", autopct='%1.1f%%', ax=axes[1], startangle=90, colors=colors)
    axes[1].set_ylabel("")  # Hide the y-axis label for the pie chart

    fig.suptitle("TrEMBL Dataset Distribution", fontsize=16)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("datasets/bacteria_inference/TrEMBL data distribution.png", format='png', dpi=300)  # You can change the filename and format

    plt.show()
     


if __name__ == "__main__":
    # dataset_dir = "datasets/TREMBL(UNIPROT)/Bacteria/"
    # fasta_file =[]
    # for item in os.listdir(dataset_dir):
    #     fasta_file.append(dataset_dir+item)
    
    # df = fasta_2_csv_uniprot_keyword(fasta_file, "datasets/TREMBL(UNIPROT)/Bacteria/")
    # print(df)
    
    # # df1 = pd.read_csv("datasets/Swissprot/Swissprot_1008_5_class.csv")
    # df1 = df
    # check_labels_count(df1)
    # df2 = sequences_family_drop(df1)
    # check_labels_count(df2)
    # df3, inference_df = drop_inference_data(df2, bacteria_list)
    # check_labels_count(df3)
    # df4 = drop_length(df3)
    # check_labels_count(df4)

    # print(inference_df)
    # check_labels_count(inference_df)
    
    # df4.to_csv(dataset_dir+"TREMBL_1008_5_class_trainable.csv", index=False)
    # inference_df.to_csv(dataset_dir+"TREMBL_1008_5_class_inference.csv", index=False)
    # print("Label Count for Swissprot data")
    # trembl_df = pd.read_csv("datasets/TREMBL(UNIPROT)/Bacteria/TREMBL_1008_5_class_trainable.csv")
    # print(len(trembl_df))
    # # TREMBL_df = pd.read_csv("env_project/datasets/TREMBL(UNIPROT)/Bacteria/TREMBL_1008_5_class_trainable.csv")
    # # check_labels_count(TREMBL_df)
    # # print(Swissprot_df.head())

    # # df = pd.read_csv("datasets_bacteria_own_keywords/new_HMP_Swissprot/Swissprot_trainable.csv")
    # # df["data type"] = "Swiss"
    # # print(len(df))

    # print(trembl_df.columns)
    # print("adskjghfadskhjldsfhljkadfs")
    

    # # check_labels_count(df)
    # # df_normal = merge_Swissprot_Keyword(Swissprot_df, df)
    # # print(len(df_normal))
    # # print(df_normal.columns)
    # # df_normal = df_normal[df_normal["label"] == 0]
    # # check_labels_count(df_normal)
    
    # # df_total = pd.concat([df_normal, Swissprot_df])    
    # check_labels_count(trembl_df)
    # #change label names
    # label_mapping = {'Secr': 1, 'AMR': 2, 'toxin': 3, 'antitox': 4, 'virul': 5, 'kill': 6, 'invasion': 7, 'phage': 8, 'effector': 9, 'putative': 10, 'hypothetical': 11}
    # trembl_df['label'] = trembl_df['label'].replace(label_mapping)
    # # df.loc[df['label'] == "Secr"] = {'Secr':"1"} 
    # # df.loc[df['label'] == "AMR"] = {'AMR':"2"} 
    # # df.loc[df['label'] == "toxin"] = {'toxin':"3"} 
    # # df.loc[df['label'] == "antitox"] = {'antitox':"4"} 
    # # df.loc[df['label'] == "virul"] = {'virul':"5"} 
    # check_labels_count(trembl_df)
    # trembl_df.to_csv("datasets/TREMBL(UNIPROT)/Bacteria/TREMBL_1008_5_class_trainable.csv", index=False)
    
# 0:normal(else) / 1:secretion / 2: resistance / 3: toxin / 4: anti-toxin / 5:virulence / 6: kill / 7: invasion / 8:phage / 9: effector / 10: putative / 11: hypothetical, unknown, uncharacterized
    df = pd.read_csv("datasets/Swissprot/Swissprot_1008_6_class_trainable.csv")
    check_labels_count(df)
    df1 = split_data(df)
    check_labels_count(df1)
    df1 = give_tensor_id(df1)
    dataset_split(df1, "datasets_swissprot/")
    # df = pd.read_csv("datasets/Swissprot/Swissprot_1008_5_class_trainable.csv")
    # df = give_tensor_id(df)
    # df.to_csv("datasets/TREMBL(UNIPROT)/Bacteria/TREMBL_1008_5_class_trainable.csv", index=False)
    # label_mapping = {
    # 1: "secretion",
    # 2: "resistance",
    # 3: "toxin",
    # 4: "anti-toxin",
    # 5: "virulence"
    # }
    # # Replace labels in the "label" column
    # df["label"] = df["label"].replace(label_mapping)
    # check_labels_count(df)
    # plot_check_labels_count(df)
    # # df_1 = split_data(df)
    
    # # print(df_1)
    # # df2 = give_tensor_id(df_1)
    # # dataset_split(df2, "datasets/Uniprot_keyword_1017_100/")