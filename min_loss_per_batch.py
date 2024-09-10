import matplotlib.pyplot as plt
import os
import numpy as np

results_path = "results/"
file_list = os.listdir(results_path)
# print(file_list)
filtered_list = [item for item in file_list if "0715" in item]
sorted_filtered_list = sorted(filtered_list)

batch_best_losses = []
for item in sorted_filtered_list:
    pth_list = os.listdir(results_path+item+"/")
    filtered_list = [item for item in pth_list if "loss" in item]
    best_losses = []
    for loss in filtered_list:
        best_loss = loss.split("_")[2].split(".")[1]
        best_losses.append(best_loss)
    print(best_losses)
    batch_best_losses.append(min(best_losses))
    

print(batch_best_losses)

# Convert the list of strings to a list of integers
# values = [int(value) for value in batch_best_losses[:-1]]
values = [int(value) for value in batch_best_losses]

# Create a list of indices for the x-axis
x = range(len(values))
new_values = np.divide(values,10000)
# Plot the values
plt.plot(x, new_values)

# Add title and labels
plt.title('Min loss per batch')
plt.xlabel('batch no')
plt.ylabel('loss')

# Show grid
plt.grid(True)

# Display the plot
# plt.show()
save_path = "train_progress/Min_loss_per_batch.png"
plt.savefig(save_path)
# plt.show()



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

# plt.savefig(save_path)
