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

class_list = ["uncha", "unknown", "hypothe", "kill", "inva", "virul", "toxi", "resi", "secre", "phage", "puta", "effector" ]
# takes fasta file as input, outputs species, id, description, sequence, length of sequence in csv
def fasta_2_csv(fasta_path, save_path):
    
    print("reading fasta file...\n-------")
    fasta_data= SeqIO.parse(open(fasta_path), "fasta")
    species_strand_list = []
    id_list = []
    seq_list = []
    des_list = []
    seq_len_list = []
    
    species_strand_list_other = []
    id_list_other = []
    seq_list_other = []
    des_list_other = []
    seq_len_list_other = []
    
    for fasta in fasta_data:
        for item in class_list:
            if item in fasta.description:
                id_list.append(fasta.id)
                seq_list.append(str(fasta.seq))
                des_list.append(fasta.description)
                seq_len_list.append(len(fasta.seq))
                match = re.search(r'OS=(.*?) OX=', fasta.description)
                species_strand_list.append(match.group(1))
            
            else:
                id_list_other.append(fasta.id)
                seq_list_other.append(str(fasta.seq))
                des_list_other.append(fasta.description)
                seq_len_list_other.append(len(fasta.seq))
                match = re.search(r'OS=(.*?) OX=', fasta.description)
                species_strand_list_other.append(match.group(1))
                
    print("read fasta file and now saving to df...\n-------")
    df = pd.DataFrame([x for x in zip(species_strand_list,
                                                id_list, des_list, seq_list, seq_len_list)],
                                columns=['species and strand', 'id', 'description',
                                        'sequence', 'length of sequence'])
    
    df.to_csv(save_path+"save.csv", index=False)
    df1 = pd.DataFrame([x for x in zip(species_strand_list_other,
                                                id_list_other, des_list_other, seq_list_other, seq_len_list_other)],
                                columns=['species and strand', 'id', 'description',
                                        'sequence', 'length of sequence'])
    # df1 = pd.DataFrame([x for x in zip(id_list_other)],
    #                             columns=['id'])
    df1.to_csv(save_path+"save_others.csv", index=False)
    
    print("saved fasta to df format!\n-------")
    return df

def fasta_2_csv_all(fasta_path, save_path):
    
    print("reading fasta file...\n-------")
    fasta_data= SeqIO.parse(open(fasta_path), "fasta")
    species_strand_list = []
    id_list = []
    seq_list = []
    des_list = []
    seq_len_list = []
    
    for fasta in fasta_data:
        id_list.append(fasta.id)
        seq_list.append(str(fasta.seq))
        des_list.append(fasta.description)
        seq_len_list.append(len(fasta.seq))
        match = re.search(r'OS=(.*?) OX=', fasta.description)
        species_strand_list.append(match.group(1))

                
    print("read fasta file and now saving to df...\n-------")
    df = pd.DataFrame([x for x in zip(species_strand_list,
                                                id_list, des_list, seq_list, seq_len_list)],
                                columns=['species and strand', 'id', 'description',
                                        'sequence', 'length of sequence'])
    
    df.to_csv(save_path+"save.csv", index=False)
    print("saved fasta to df format!\n-------")
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
    
#drops rows longer than 1024
def drop_length(df):
    print("removing rows longer than 1024...\n-------")
    df1 = df[df["length of sequence"].astype(int)<=1024]
    print("removed rows longer than 1024!\n-------")
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

# # df_new.to_csv("results/fungi_32_species/balanced_4606_peft/fungi_1024_3class_32organism_inference_results_ALL.csv", index=False)

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
    kill_df = df_by_labels.get_group(6)
    invasion_df = df_by_labels.get_group(7)
    phage_df = df_by_labels.get_group(8)
    effector_df = df_by_labels.get_group(9)
    putative_df = df_by_labels.get_group(10)
    # unknown_df = df_by_labels.get_group(11)
    
    
    normal_random_df = normal_df.sample(100)
    secretion_random_df = secretion_df.sample(100)
    resistant_random_df = resistant_df.sample(100)
    toxin_random_df = toxin_df.sample(100)
    # putatuve_random_df = putative_df.sample(14006)
    # phage_random_df = phage_df.sample(14006)
    anti_toxin_random_df = anti_toxin_df.sample(100)
    virulence_random_df = virulence_df.sample(100)
    # unknow_random_df = unknown_df.sample(14006)
    # # sample_df = pd.concat([normal_random_df, secretion_random_df, resistant_random_df, toxin_random_df, virulence_df, invasion_df, 
    #                        kill_df, phage_random_df,anti_toxin_random_df])
    
    sample_df = pd.concat([normal_random_df, secretion_random_df, resistant_random_df, toxin_random_df, anti_toxin_random_df, virulence_random_df])
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
        

def merge_Swissprot_HMP(df, df1):
    
    df_all = pd.concat([df, df1])
    df_all = df_all.drop_duplicates(subset=['sequence'], keep="last")
    
    return df_all


swissprot_path = root+"SwissProt(UNIPROT)/uniprot_bacteria_1224.fasta"
HMP_path = root + "HMP/HMP_1024_ALL.csv"
bacteria_save_path = root+"SwissProt(UNIPROT)/"

def swiss_prot_trainable():
    df = fasta_2_csv_all(swissprot_path, bacteria_save_path)
    df = pd.read_csv("datasets/SwissProt(UNIPROT)/save.csv")
    # print(df.head())
    # df1 = label_data_bacteria(df)
    # df2 = sequences_family_drop(df1)
    # df3 = drop_length(df2)
    # df4, inference_df = drop_inference_data(df2, bacteria_list)
    # check_labels_count(df4)
    # print(inference_df)
    # df4.to_csv("datasets/new_HMP_Swissprot/Swissprot_trainable.csv", index=False)
    # inference_df.to_csv("datasets/new_HMP_Swissprot/Swissprot_inference.csv", index=False)
    
    
if __name__ == "__main__":
    # df = pd.read_csv("datasets/SwissProt(UNIPROT)/save.csv")
    # print(df.head())
    # print("--------------------------------------------------")
    # df1 = pd.read_csv("datasets/new_HMP/HMP_0403.csv")
    # print(df.head())
    
    # # make new swiss prot trainable data!
    # swissprot_path = "datasets/bacteria_inference/Streptococcus_mutans_swissprot_0613.fasta"
    # bacteria_save_path = "datasets/bacteria_inference/"
    # df = fasta_2_csv_all(swissprot_path, bacteria_save_path)
    # df = pd.read_csv("datasets/bacteria_inference/save.csv")
    # df1 = label_data_bacteria(df)
    # df2 = sequences_family_drop(df1)
    # df3 = drop_length(df2)
    # df4, inference_df = drop_inference_data(df2, bacteria_list)
    # index_no = []
    # for i in range(len(df4["id"])):
    #    index_no.append(i)
    # df4["tensor id"] = index_no
    # check_labels_count(df4)
    # print(inference_df)
    # df4.to_csv("datasets/bacteria_inference/Streptococcus_mutans_swissprot.csv", index=False)
    # inference_df.to_csv("datasets/new_HMP_Swissprot/Swissprot_inference.csv", index=False)
    
    # df = pd.read_csv("datasets/HMP.csv")
    # check_labels_count(df)
    # df = df.iloc[:, [4,5,6,7,8]]
    # df.rename(columns = {'bacteria species':'species and strand'}, inplace = True) 
    # df.to_csv("datasets/new_HMP/HMP_0403.csv", index=False)
    
    # df = pd.read_csv("datasets/new_HMP_Swissprot/HMP_trainable.csv")
    # print("below is HMP data")
    # print(df)   
    # check_labels_count(df)
    # print("==========================")
    
    
    # df1 = pd.read_csv("datasets/new_HMP_Swissprot/Swissprot_trainable.csv")
    # print("below is Swissprot data")
    # print(df1)
    # check_labels_count(df1)
    # df_all = df1.drop_duplicates(subset=['sequence'], keep="last")
    # check_labels_count(df_all)
    # print(len(df)+len(df1))
    # HMP_swissprot_df = merge_Swissprot_HMP(df, df1)
    # print(len(HMP_swissprot_df))
    # print(HMP_swissprot_df)
    # HMP_swissprot_df.to_csv("datasets/new_HMP_Swissprot/HMP_Swissprot_0403.csv", index = False)
    # check_labels_count(HMP_swissprot_df)
    
    # df = pd.read_csv("datasets/sample_0216/train.csv")
    # check_labels_count(df)
    # df = pd.read_csv("datasets/sample_0216/test.csv")
    # check_labels_count(df)
    # df = pd.read_csv("datasets/sample_0216/valid.csv")
    # check_labels_count(df)
    df1 = pd.read_csv("datasets/new_HMP_Swissprot/HMP_Swissprot_0403.csv")
    
    df_sample = split_data(df1)
    index_no = []
    for i in range(len(df_sample["id"])):
       index_no.append(i)
    df_sample["tensor id"] = index_no
    df_sample.to_csv("datasets/Batch_train_start/HMP_Swissprot_6class_100_0701.csv", index = False)
    print(df_sample)
    check_labels_count(df_sample)
    dataset_split(df_sample, "datasets/HMP_Swissprot_0701_100/")
    
    
    # df1 = pd.read_csv(tremble_save_path+"Tremble_trainable.csv")
    # batch_training_data(df1)
    # print(len(df1))
    
    # df2 = pd.read_csv("datasets/new_HMP/HMP_new_1024_ALL.csv")
    # df2 = label_data_bacteria(df2)
    # # df2 = sequences_family_drop(df2)
    # df2 = drop_length(df2)
    # df2, inference_df = drop_inference_data(df2, bacteria_list)
        # df3 = pd.read_csv("datasets/swissprot_new_1024_ALL_1224.csv")
    # df3 = label_data_bacteria(df3)
    # # df3 = sequences_family_drop(df3)
    # df3 = drop_length(df3)
    # df3, inference_df = drop_inference_data(df3, bacteria_list)
    # HMP_df = df2[["id", "description", "sequence", "bacteria species", "label"]]
    # SWISS_df = df3[["id", "description", "sequence", "bacteria species", "label"]]
    # print(HMP_df)
    # print(SWISS_df)
    
    # df_all = pd.concat([HMP_df, SWISS_df, df1])
    # df_all = df_all.drop_duplicates(keep=False)
    # print(df_all)
    # check_labels_count(df1)
    # df_sample = split_data(df_all)


    # index_no = []pl
    # for i in range(len(df_sample["id"])):
    #    index_no.append(i)
    # df_sample["tensor id"] = index_no
        
    # dataset_split(df_sample, "datasets/sample_0216/")
    
    
    # for group_fa in os.listdir(tremble_group_path):
    #     print(group_fa)
    #     df = fasta_2_csv(tremble_path, bacteria_save_path)
        
    #     print(df)
        # break
    
    # df = fasta_2_csv(tremble_path, bacteria_save_path)
    
    # print("done")
    # df1 = label_data_bacteria(df)
    # df2 = sequences_family_drop(df1)
    # df3 = drop_length(df2)
    # df4, inference_df = drop_inference_data(df3, bacteria_list)
    # check_labels_count(df1)
    # HMP_df = pd.read_csv("datasets/new_HMP_Swissprot/HMP_new_1024_ALL_balanced.csv")
    # Swissprot = pd.read_csv("datasets/new_HMP_Swissprot/swissprot_new_1024_ALL_1224.csv")
    
    # print(HMP_df)
    # check_labels_count(HMP_df)
    # HMP_df = HMP_df[HMP_df["label"] != 5]
    # new_HMP_df = HMP_df[HMP_df["label"] != 6]
    # check_labels_count(new_HMP_df)
    
    # dataset_split(new_HMP_df, "datasets/new_HMP_Swissprot/5_class")
    
    # df = pd.read_csv("datasets/bacteria_inference/Salmonella_typhimurium_(strainLT2).csv")
    # df_compare = pd.read_csv("results/240202_processed_data/Salmonella_typhimurium_(strainLT2)_0302.csv")
    
    # df_new = merge_inferenece(df, df_compare)
    # df_new.to_csv("results/240202_processed_data/Salmonella_typhimurium_(strainLT2)_0302.csv", index=False)
    # 0:normal(else) / 1:secretion / 2: resistance / 3: toxin / 4: anti-toxin / 5:virulence / 6: invasion / 7: kill / 8:phage / 9: putative / 10: hypothetical, unknown, uncharacterized
 
    # df = pd.read_csv("results/240202_processed_data/Salmonella_typhimurium_(strainLT2)_0302.csv")
    
    # df["max"] = df[['normal', 'secretion', 'resistance', 'toxin', 'anti-toxin', 'virulence', 'invasion', 'kill']].idxmax(axis=1)
    # print(df)
    # df.to_csv("results/240202_processed_data/Salmonella_typhimurium_(strainLT2)_0302.csv", index=False)
    # df_2 = df[df["toxin"] > 0.4]
    # df_possible_toxin = df_2[df_2["toxin"]<0.6]
    # print(df_possible_toxin)
    
    # df_2 = df[df["secretion"] > 0.4]
    # df_possible_secretion = df_2[df_2["secretion"]<0.6]
    # print(df_possible_secretion)
    
    # df_2 = df[df["resistance"] > 0.4]
    # df_possible_resistance = df_2[df_2["resistance"]<0.6]
    # print(df_possible_resistance)
    # print("FUCL")
    
    
    
    # # df_possible_virulence = df[df["virulence"] > 0.1]
    # # df_possible_secretion = df[df["secretion"] > 0.3]
    # # df_possible_resistance = df[df["resistance"] > 0.3]
    
    # # df_possible_toxin.to_csv("results/new_HMP_Swissprot/PEFT_5_class/Salmonella_typhimurium_(strainLT2)_0108_possible_toxin.csv", index=False)
    # # df_possible_secretion.to_csv("results/new_HMP_Swissprot/PEFT_5_class/Salmonella_typhimurium_(strainLT2)_0108_possible_secretion.csv", index=False)
    # # df_possible_resistance.to_csv("results/new_HMP_Swissprot/PEFT_5_class/Salmonella_typhimurium_(strainLT2)_0108_possible_resistance.csv", index=False)
    
    # df3 = pd.concat([df_possible_toxin, df_possible_secretion, df_possible_resistance])
    # df3 = df3.drop_duplicates(keep=False)
    # print(df3)
    # df3.to_csv("results/new_HMP_Swissprot/PEFT_5_class/Salmonella_typhimurium_(strainLT2)_0108_possible_2class.csv", index=False)
    
    
# 0:normal(else) / 1:secretion / 2: resistance / 3: toxin / 4: anti-toxin / 5:virulence / 6: kill / 7: invasion / 8:phage / 9: effector / 10: putative / 11: hypothetical, unknown, uncharacterized

    # df = pd.read_csv("datasets/new_HMP_Swissprot/HMP_Swissprot_0403.csv")
    # print(df)
    # check_labels_count(df)
    # df1 = label_data_bacteria(df)
    # check_labels_count(df1)
    
    # df_sample = split_data(df1)
    # index_no = []
    # for i in range(len(df_sample["id"])):
    #    index_no.append(i)
    # df_sample["tensor id"] = index_no
    # check_labels_count(df_sample)
    
    # dataset_split(df_sample, "datasets/HMP_Swissprot_0508/")
    
    
    # df = pd.read_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0.csv")
    # index_no = []
    # for i in range(len(df["id"])):
    #    index_no.append(i)
    # df["tensor id"] = index_no
    # df = label_data_bacteria(df)
    # check_labels_count(df)
    # df = df[df.label != 9]
    # df = df[df.label != 8]
    # df = df[df.label != 7]
    # df = df[df.label != 6]
    # check_labels_count(df)
    # df.to_csv("datasets/TremBLE(UNIPROT)/Bacteria/batch_data/batch_0_5class.csv", index=False)
    