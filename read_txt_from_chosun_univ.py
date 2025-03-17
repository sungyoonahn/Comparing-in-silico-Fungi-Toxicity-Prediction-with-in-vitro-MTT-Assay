import pandas as pd
import os
import re


def read_txt(file_path):
    with open(file_path, 'r') as file:
        return file.read()

txt_file = read_txt("datasets/bacteria_inference/chosun_univ_txt/LT2-genome-total.txt")

# Extract CDS blocks
cds_blocks = re.findall(r'CDS.*?(?=\n     \S|$)', txt_file, re.DOTALL)

# Prepare lists for DataFrame
ids = []
descriptions = []
sequences = []
genes = []
# Extract required details from each CDS block
for block in cds_blocks:
    protein_id = re.search(r'/protein_id="(.*?)"', block)
    product = re.search(r'/product="(.*?)"', block)
    translation = re.search(r'/translation="(.*?)"', block, re.DOTALL)
    gene = re.search(r'/gene="(.*?)"', block, re.DOTALL)

    ids.append(protein_id.group(1) if protein_id else "Not Found")
    descriptions.append(product.group(1) if product else "Not Found")
    genes.append(gene.group(1) if gene else "Not Found")

    if translation:
        translation_text = translation.group(1).replace('\n', '').replace(' ', '')
        sequences.append(translation_text)
    else:
        sequences.append("Not Found")

# Create DataFrame
df = pd.DataFrame({
    'id': ids,
    'description': descriptions,
    'gene':genes,
    'sequence': sequences
})

# Display DataFrame
print(df)

df["label"] = [0]*len(df)
index_no = []
for i in range(len(df)):
   index_no.append(i)
df["tensor id"] = index_no


df.to_csv("datasets/bacteria_inference/chosun_univ_txt/LT2-genome-total.csv", index=False)

df = pd.read_csv("datasets/bacteria_inference/chosun_univ_txt/LT2-genome-total.csv")
print(df)