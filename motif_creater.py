# import pandas as pd

# def write_fasta(sequences, file_name):
#     with open(file_name, 'w') as fasta_file:
#         for i, sequence in enumerate(sequences, 1):
#             fasta_file.write(f'>sequence_{i}\n')
#             fasta_file.write(f'{sequence}\n')

# # Example list of protein sequences
# sequences = [
#     'MVLSPADKTNVKAAW',
#     'DLGPRRDRFSLHLL',
#     'GLSDGEWQQVLNVW',
# ]

# # Write the protein sequences to a FASTA file
# write_fasta(sequences, 'protein_sequences.fasta')



import os
from collections import defaultdict
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import numpy as np
import warnings
from Bio import SeqIO
warnings.simplefilter(action='ignore', category=FutureWarning)

class_name = "anti-toxin"

### run below code in terminal
## 1. groups all fasta files to one fasta file
# cat *.fasta > virulence_motifs.fasta

## 2. clusters sequences using mmseq2 and saves to clusterRes_cluster.tsv
# mmseqs easy-cluster virulence_motifs.fasta clusterRes tmp

# # 3. Below code is for reading the cluster data
# def read_fasta(fasta_file):
#     """Reads a FASTA file and returns a dictionary of sequence ID -> sequence."""
#     sequences = {}
#     with open(fasta_file, 'r') as file:
#         seq_id = ""
#         sequence = ""
#         for line in file:
#             line = line.strip()
#             if line.startswith(">"):
#                 if seq_id:
#                     sequences[seq_id] = sequence
#                 seq_id = line[1:]  # Remove the '>'
#                 sequence = ""
#             else:
#                 sequence += line
#         if seq_id:
#             sequences[seq_id] = sequence
#     return sequences

# def save_to_fasta(sequences, output_file):
#     """Save sequences to a FASTA file."""
#     with open(output_file, 'w') as file:
#         for seq_id, sequence in sequences.items():
#             file.write(f">{seq_id}\n")
#             file.write(f"{sequence}\n")

# # Step 1: Read the cluster output
# clusters = defaultdict(list)

# with open(f'results/attention_visualized/{class_name}/clusterRes_cluster.tsv', 'r') as file:
#     for line in file:
#         representative, sequence_id = line.strip().split('\t')
#         clusters[representative].append(sequence_id)

# # Step 2: Sort clusters by size and select top 5
# sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:10]

# # Step 3: Read sequences from the original FASTA file
# fasta_sequences = read_fasta(f'results/attention_visualized/{class_name}/{class_name}_motifs.fasta')

# # Step 4: Save the sequences for the top 5 clusters
# for i, (representative, sequence_ids) in enumerate(sorted_clusters):
#     output_sequences = {}
#     for seq_id in sequence_ids:
#         if seq_id in fasta_sequences:
#             output_sequences[seq_id] = fasta_sequences[seq_id]
#         else:
#             print(f"Sequence {seq_id} not found in the FASTA file.")
    
#     # Save the sequences for the current cluster to a FASTA file
#     output_file = f"results/attention_visualized/{class_name}/cluster_{i+1}_sequences.fasta"
#     save_to_fasta(output_sequences, output_file)
#     print(f"Saved cluster {i+1} sequences to {output_file}")
    
    
# 4. sequences align fasta files
# clustalo -i cluster_1_sequences.fasta -o cluster_1_aligned_sequences.fasta --outfmt=fasta
# clustalo -i cluster_2_sequences.fasta -o cluster_2_aligned_sequences.fasta --outfmt=fasta
# clustalo -i cluster_3_sequences.fasta -o cluster_3_aligned_sequences.fasta --outfmt=fasta
# clustalo -i cluster_4_sequences.fasta -o cluster_4_aligned_sequences.fasta --outfmt=fasta
# clustalo -i cluster_5_sequences.fasta -o cluster_5_aligned_sequences.fasta --outfmt=fasta
# clustalo -i cluster_6_sequences.fasta -o cluster_6_aligned_sequences.fasta --outfmt=fasta
# clustalo -i cluster_7_sequences.fasta -o cluster_7_aligned_sequences.fasta --outfmt=fasta
# clustalo -i cluster_8_sequences.fasta -o cluster_8_aligned_sequences.fasta --outfmt=fasta
# clustalo -i cluster_9_sequences.fasta -o cluster_9_aligned_sequences.fasta --outfmt=fasta
# clustalo -i cluster_10_sequences.fasta -o cluster_10_aligned_sequences.fasta --outfmt=fasta


# # 5. create logos
for i in range(10):
    fasta_data= SeqIO.parse(open(f"results/attention_visualized/{class_name}/cluster_{i+1}_aligned_sequences.fasta"), "fasta")

    sequences = []
    for fasta in fasta_data:
        sequences.append(str(fasta.seq))

    counts_mat = logomaker.alignment_to_matrix(sequences=sequences, to_type='counts', characters_to_ignore='.-X')

    print(counts_mat.head())

    nn_logo = logomaker.Logo(counts_mat, color_scheme='chemistry')
    plt.savefig(f"results/attention_visualized/{class_name}/cluster_{i+1}_aligned_sequences.jpg", dpi=300, bbox_inches='tight')
    
    