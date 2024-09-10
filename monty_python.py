from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import AlignIO
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from Bio.Blast.Applications import NcbiblastpCommandline

from collections import Counter

class_name = "toxin"

def find_most_likely_motif(alignment, threshold=0.8):
    """
    Find the most conserved motif in a Clustal FASTA alignment.
    
    :param alignment: Multiple sequence alignment object
    :param threshold: Fraction of sequences that must have the same amino acid at a position for it to be considered conserved
    :return: Consensus sequence (most likely motif)
    """
    num_sequences = len(alignment)
    motif = []

    # Loop over columns (positions in the alignment)
    for i in range(alignment.get_alignment_length()):
        column = alignment[:, i]
        
        # Count the frequency of each amino acid at this position (ignoring gaps '-')
        counts = Counter(column.replace("-", ""))
        
        if counts:
            # Find the most common amino acid
            most_common_aa, frequency = counts.most_common(1)[0]
            
            # Check if it meets the conservation threshold
            if frequency / num_sequences >= threshold:
                motif.append(most_common_aa)
            else:
                motif.append("-")  # Less conserved position
        else:
            motif.append("-")  # Gaps in all sequences

    return "".join(motif)



motifs =[]
for i in range(5):
    # Load the aligned sequences from a Clustal FASTA file
    alignment = AlignIO.read(f"results/attention_visualized/{class_name}/cluster_{i+1}_aligned_sequences.fasta", "fasta")
    # Find and print the most likely motif (based on a 80% conservation threshold)
    most_likely_motif = find_most_likely_motif(alignment, threshold=0.8)
    first_index = next((i for i, char in enumerate(most_likely_motif) if char != '-'), None)
    last_index = next((i for i in range(len(most_likely_motif) - 1, -1, -1) if most_likely_motif[i] != '-'), None)
    trimmed_motif = most_likely_motif[first_index:last_index + 1]
    print(f"Most likely motif:{trimmed_motif}")

    # Convert motif to regex
    re_pattern = trimmed_motif.replace('-', '.*')
    # Compile the regex
    motif = re.compile(re_pattern)
    motifs.append(motif)

print(motifs)





# Your sequences
salmonela_df = pd.read_csv("results/bacteria_inference/Salmonella_typhimurium_(strainLT2)0806.csv")
print(len(salmonela_df))
salmonela_anti_toxin_df = salmonela_df[salmonela_df["max label"] == "toxin"]
sequences = []
for record in salmonela_anti_toxin_df["sequence"]:
    sequences.append("".join(record.split()))
print(len(sequences))

matched_sequences = []
for motif in tqdm(motifs):
    matched_sequences.extend([seq for seq in tqdm(sequences) if motif.search(seq)])

# Remove duplicates, if necessary
matched_sequences = list(set(matched_sequences))

# If there are matching sequences, run a BLAST search
if matched_sequences:
    print("Matched Sequences:")
    for seq in matched_sequences:
        print(seq)

    # Save matched sequences to a FASTA file for BLAST
    with open("test.fasta", "w") as f:
        for i, seq in enumerate(matched_sequences):
            f.write(f">seq{i+1}\n{seq}\n")

    # Run BLAST
    blastp_cline = NcbiblastpCommandline(
        query="test.fasta",
        db="datasets/Swissprot/swissprot_bacteria_db",  # or your local database name
        evalue=0.001,
        outfmt=5,
        out="blast_results.xml"
    )

    stdout, stderr = blastp_cline()
    print("BLAST search completed. Results saved in blast_results.xml")
else:
    print("No sequences matched any of the motifs.")