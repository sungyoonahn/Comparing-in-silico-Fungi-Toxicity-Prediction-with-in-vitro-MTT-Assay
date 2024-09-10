import pandas as pd
from Bio import SeqIO
import os
import re
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
"Salmonella typhimurium (strain LT2 / SGSC1412 / ATCC 700720)"]


print("List of all the species of bacteria")
print(len(bacteria_list))

species_list = []
for item in bacteria_list:
    species = item.split(" ")[0]
    species_list.append(species)

family_list = set(species_list)
print("List of all the family of bacteria")
print(family_list)
print(len(family_list))


# {'Mammaliicoccus',
# 'Planomicrobium',
# '[Brevibacterium]', 'Bacillus', 'Pseudomonas', 'Leucobacter', 'Kocuria', 'Paenarthrobacter', 'Brevundimonas', 'Dermacoccus', 'Staphylococcus', 'Corynebacterium', 'Dietzia', 'Brachybacterium', 'Lysinibacillus',
#  'Micrococcus', 'Roseomonas', 'Exiguobacterium', 'Moraxella', 'Brevibacterium', 'Microbacterium', 'Chryseobacterium', 'Acinetobacter', 'Rhodococcus', 'Psychrobacillus', 'Agrococcus', 'Priestia',
#  'Rosenbergiella', 'Arthrobacter', 'Paenibacillus', 'Nocardia', 'Massilia', 'Streptomyces', 'Salmonella', 'Neomicrococcus', 'Pseudarthrobacter'}

root_path = "datasets/new_uniprot_bacteria/70family.fa"

id_list = []
seq_list = []
des_list = []
seq_len_list = []
species_strand_list = []

fasta_sequences = SeqIO.parse(open(root_path), "fasta")
for fasta in fasta_sequences:
    # target_organ_list.append(item)
    # bacteria_name_list.append(bacteria_name)
    id_list.append(fasta.id)
    seq_list.append(str(fasta.seq))
    des_list.append(fasta.description)
    seq_len_list.append(len(fasta.seq))
    match = re.search(r'OS=(.*?) OX=', fasta.description)
    species_strand_list.append(match.group(1))
    print(fasta.id)


df = pd.DataFrame([x for x in zip(id_list, des_list, seq_list, seq_len_list, species_strand_list)],
                                  columns=['id', 'description', 'sequence', 'length of sequence', "species and strand"])


print(df)

species_list = []

for species in df["species and strand"]:
    split_string = str(species).split(" ")
    if len(split_string) == 1:
        species_list.append("nan")
    else:
        combined_string = split_string[0] + " " + split_string[1]
        print(combined_string)
        species_list.append(combined_string)

    print(species)

df["species"] = species_list
print("before removing proteins with only family")
print(len(df))
df_1 = df[df["species"] != "nan"]
print("after removing proteins with only family")
print(len(df_1))

test_df = pd.DataFrame(columns=['id', 'description', 'sequence', 'length of sequence', "species and strand", "species"])

for species_name in bacteria_list:
   contain_values = df[df["species"].str.contains(species_name)]
   test_df = pd.concat([test_df, contain_values])

print(df)
test_df.to_csv("datasets/new_uniprot_bacteria/70family.csv", index=False)



