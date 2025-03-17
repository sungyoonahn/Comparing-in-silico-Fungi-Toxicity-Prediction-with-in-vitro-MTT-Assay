import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd



def plot_ACC_graph(y_values, best_ACCS, title, y_label, save_path):
    # Create a list of indices for the x-axis
    # x = range(len(best_ACCS))
    new_values = np.multiply(best_ACCS,100)
    # Plot the values
    plt.plot(y_values, new_values, color='skyblue')
    # plt.xticks(y_values)
    # Add title and labels
    plt.title(title)
    plt.xlabel('Data Count')
    plt.ylabel(y_label)

    # Show grid
    plt.grid(True)

    # Display the plot
    # plt.show()
    plt.savefig(save_path)

def plot_graph(y_values, best_values, title, y_label, save_path):
    # Create a list of indices for the x-axis
    # x = range(len(best_values))
    # Plot the values
    plt.plot(y_values, best_values, color='skyblue')
    # plt.xticks(y_values)
    # Add title and labels
    plt.title(title)
    plt.xlabel('Data Count')
    plt.ylabel(y_label)

    # Show grid
    plt.grid(True)

    # Display the plot
    # plt.show()
    plt.savefig(save_path)

results_path = "results/"
file_list = os.listdir(results_path)
# print(file_list)
filtered_list = [item for item in file_list if "1017" in item]
sorted_filtered_list = sorted(filtered_list)

batch_best_losses = []
best_ACCS = []
best_MCCS = []
best_F1S = []
best_AUROCS = []
y_values = []
for item in sorted_filtered_list:
    pth_list = os.listdir(results_path+item+"/")
    y_values.append(int(item.split("_")[2]))
    df = pd.read_csv(results_path+item+"/results.csv")
    dfs = [df[i:i+4] for i in range(0, len(df), 4)]
    # print(df.head())
    
    filtered_list = [item for item in pth_list if "loss" in item]
    best_losses = []
    for loss in filtered_list:
        best_loss = loss.split("_")[2].split(".")[1]
        best_epoch = loss.split("_")[0]
        best_losses.append(best_loss)
    # get ACC of best loss testset
    for df in dfs:
        if df["epoch"][df.index.values[0]] == best_epoch:
            best_ACC = df["ACC"][df.index.values[3]]
            best_MCC = df["MCC"][df.index.values[3]]
            best_F1 = df["F1"][df.index.values[3]]
            best_AUROC = df["AUROC"][df.index.values[3]]
        else:
            continue

    print(best_ACC)
    print(best_losses)
    batch_best_losses.append(min(best_losses))
    best_ACCS.append(best_ACC)
    best_MCCS.append(best_MCC)
    best_F1S.append(best_F1)
    best_AUROCS.append(best_AUROC)
    
    

print(batch_best_losses)
print(y_values)
# print(best_ACCS)

# values = [int(value) for value in batch_best_losses]

# # Create a list of indices for the x-axis
# # x = range(len(values))
# new_values = np.divide(values,10000)
# # Plot the values
# plt.plot(y_values, new_values, color='skyblue')
# # plt.xticks(y_values)
# # Add title and labels
# plt.title('Minimum Loss per Data Count')
# plt.xlabel('Data Count')
# plt.ylabel('Loss')

# # Show grid
# plt.grid(True)

# save_path = "train_progress/Min_loss_per_batch_uniprot_keyword.png"
# plt.savefig(save_path)

# plot_ACC_graph(y_values, best_ACCS, "Accuracy per Data Count", "Accuracy", "train_progress/ACC_per_batch_uniprot_keyword.png")
# plot_graph(y_values, best_MCCS, "MCC per Data Count", "MCC", "train_progress/MCC_per_batch_uniprot_keyword.png")
# plot_graph(y_values, best_F1S, "F1 per Data Count", "F1", "train_progress/F1_per_batch_uniprot_keyword.png")
plot_graph(y_values, best_AUROCS, "AUROC per Data Count", "AUROC", "train_progress/AUROC_per_batch_uniprot_keyword.png")
# ## best ACC
# # values = [int(value) for value in best_ACC]

# # Create a list of indices for the x-axis
# x = range(len(best_ACCS))
# new_values = np.multiply(best_ACCS,100)
# # Plot the values
# plt.plot(y_values, new_values, color='skyblue')
# # plt.xticks(y_values)
# # Add title and labels
# plt.title('Accuracy per Data Count')
# plt.xlabel('Data Count')
# plt.ylabel('Accuracy')

# # Show grid
# plt.grid(True)

# # Display the plot
# # plt.show()
# save_path = "train_progress/ACC_per_batch_uniprot_keyword.png"
# plt.savefig(save_path)

#####PLOT DATA #####

# def join_train_data(path):
#     df_train = pd.read_csv(path+"train.csv")
#     df_valid = pd.read_csv(path+"valid.csv")
#     df_test = pd.read_csv(path+"test.csv")
#     df = pd.concat([df_train, df_valid, df_test])
    
#     return df

# def check_labels_count(df):
#     print("checking count for each labels...\n-------")
#     print(pd.Series(df["label"]).value_counts().reset_index())



# import matplotlib.pyplot as plt
# path = "datasets/HMP_Swissprot_0403/"
# # df = join_train_data(path)
# df = pd.read_csv("datasets/TremBLE(UNIPROT)/Bacteria/Tremble_trainable.csv")
# df1 = pd.Series(df["label"]).value_counts().reset_index()
# df1 = df1.sort_values(by=['label'])
# # 0:normal(else) / 1:secretion / 2: resistance / 3: toxin / 4: anti-toxin / 5:virulence / 6: kill / 7: invasion / 8:phage / 9: effector / 10: putative / 11: hypothetical, unknown, uncharacterized

# labels = ["normal", "secretion", "resistance", "toxin", "anti-toxin", "virulence"]
# # check_labels_count(df)
# df1["label"] = df1["label"].replace([0], "normal")
# df1["label"] = df1["label"].replace([1], "secretion")
# df1["label"] = df1["label"].replace([2], "resistance")
# df1["label"] = df1["label"].replace([3], "toxin")
# df1["label"] = df1["label"].replace([4], "anti-toxin")
# df1["label"] = df1["label"].replace([5], "virulence")
# print(df1)
# df1 = df1.loc[(df1['label'] == 'secretion') | (df1['label'] == 'resistance')| (df1['label'] == 'toxin')
#               | (df1['label'] == 'anti-toxin')| (df1['label'] == 'virulence')]

# df1.plot(x='label', y='count',kind="bar",title="My plot")
# plt.xticks(rotation=0)
# # plt.axis('equal')
# # plt.pie(df1["count"], labels=df1["label"], autopct='%1.1f%%')
# plt.title("TrEMBLE")
# save_path = "train_progress/tremble_train_data_bar.png"

# plt.savefig(save_path)s
