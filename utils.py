import pandas as pd
from sklearn.model_selection import train_test_split


def tensor_id_merge(df1, df2):
### MERGE INPUT DATAFRAMES BASED ON THEIR TENSOR ID
    df3 = pd.merge(df1, df2, on='tensor id', how='inner')
    return df3

def merge_inferenece(df, df_compare):
### BELOW CODE IS FOR MERGING INFERENCE RESULTS WITH THEIR INFORMATION
    print(df_compare)
    # df_compare = df_compare.drop('Unnamed: 0', axis=1)
    df_new = tensor_id_merge(df, df_compare)
    return df_new

def check_labels_count(df):
### BELOW CODE IS FOR CHECKING HOW MANY ITEMS THERE ARE IN EACH CLASS
    print("checking count for each labels...\n-------")
    print(pd.Series(df["label"]).value_counts().reset_index())

def preds_check_labels_count(df):
### BELOW CODE IS FOR CHECKING HOW MANY ITEMS THERE ARE IN EACH PREDICTED CLASS
    print("checking count for each labels...\n-------")
    print(pd.Series(df["max label"]).value_counts().reset_index())
    
def matching_labels(df):
### BELOW CODE IF FOR CHEKCING IF THE PREDICTED CLASS MATCHES THE ORIGINAL CLASS
    print("checking matching prediction and base results...'n")
    matching_labels = df["max label"] == df["label"]
    print(matching_labels)
    df["matching label"] = matching_labels
    return df

def dataset_split(df, save_dir):
### BELOW IS THE CODE FOR SPLITING THE RAW DATA INTO TRAIN/VALIDATION/TEST SETS
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
### BELOW IS THE CODE FOR ADDING ALL THE TRAIN, VALID, TEST DATA IN TO ONE DF
    df_train = pd.read_csv(path+"train.csv")
    df_valid = pd.read_csv(path+"valid.csv")
    df_test = pd.read_csv(path+"test.csv")
    df = pd.concat([df_train, df_valid, df_test])
    return df

def add_train_data(path, save_path, sample_df):
### BELOW IS THE CODE FOR ADDING TRAIN, VALID, TEST DATA TO EACH OF ITS OWN DATASETS
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
### BELOW IS THE CODE FOR DROPING ALL DUPLICATE SEQUENCES
    # Column to use for matching
    column_to_match = 'sequence'

    # Merge the DataFrames on the specified column
    merged = pd.merge(df, df_compare, on=column_to_match, how='inner')

    # Filter out the rows from df_b that are duplicates of rows in df_a based on the specified column
    result = df[~df[column_to_match].isin(merged[column_to_match])]

    return result

def sample_trainable_df_5class(df, sample_count):
### BELOW IS THE CODE IN SAMPLING THE DATA FOR BACTERIA
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
