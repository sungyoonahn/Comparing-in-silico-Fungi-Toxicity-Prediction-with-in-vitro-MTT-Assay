from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio import AlignIO
import pandas as pd
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from itertools import product

class_name = "anti-toxin"


def find_most_likely_motif(alignment, threshold=0.8):
    print(threshold)
    """
    Find all possible motifs in a Clustal FASTA alignment when there are multiple most common amino acids.
    
    :param alignment: Multiple sequence alignment object
    :param threshold: Fraction of sequences that must have the same amino acid at a position for it to be considered conserved.
    :return: List of all possible motifs
    """
    num_sequences = len(alignment)
    motif_positions = []

    # Loop over columns (positions in the alignment)
    for i in range(alignment.get_alignment_length()):
        column = alignment[:, i]

        # Count the frequency of each amino acid at this position (ignoring gaps '-')
        counts = Counter(column.replace("-", ""))
        if counts:
            # Get all amino acids that meet the threshold
            valid_aa = [aa for aa, freq in counts.items() if freq / num_sequences >= threshold]
            
            if valid_aa:
                motif_positions.append(valid_aa)
            else:
                motif_positions.append(["-"])  # Less conserved position
        else:
            motif_positions.append(["-"])  # Gaps in all sequences

    # Generate all possible motif combinations
    all_possible_motifs = ["".join(motif) for motif in product(*motif_positions)]

    return all_possible_motifs




def compile_motifs(num_clusters):
    """
    Compile motifs from multiple sequence alignments.
    
    :param num_clusters: Number of clusters to process
    :return: List of compiled regex motifs
    """
    motifs = []
    for i in range(num_clusters):
        alignment = AlignIO.read(f"results/attention_visualized/{class_name}/cluster_{i + 1}_aligned_sequences.fasta", "fasta")
        most_likely_motifs = find_most_likely_motif(alignment, threshold=0.8)
        for most_likely_motif in most_likely_motifs:
            first_index = next((i for i, char in enumerate(most_likely_motif) if char != '-'), None)
            last_index = next((i for i in range(len(most_likely_motif) - 1, -1, -1) if most_likely_motif[i] != '-'), None)
            
            trimmed_motif = most_likely_motif[first_index:last_index + 1]
            re_pattern = trimmed_motif.replace('-', '.*')  # Convert to regex pattern
            # exclude motifs that have more than 9 '*' characters
            if re_pattern.count('*') > 8:
                continue
            else:           
                motif = re.compile(re_pattern)
                motifs.append(motif)
    
    return motifs

def match_sequence_to_motifs(sequence, motifs):
    """
    Match a sequence against all motifs.
    
    :param sequence: The sequence to match
    :param motifs: List of compiled regex motifs
    :return: True if the sequence matches any motif, otherwise False
    """
    return any(motif.search(sequence) for motif in motifs)

# Load sequences
salmonela_df = pd.read_csv("results/bacteria_inference/Streptococcus_mutans_swissprot1029.csv")
print(salmonela_df.head())
salmonela_max_df = salmonela_df[salmonela_df["max label"] == f"{class_name}"]
print(len(salmonela_max_df))
salmonela_max_df = salmonela_max_df[salmonela_max_df["sequence"].str.len() <= 1024]
print(len(salmonela_max_df))
sequences = ["".join(record.split()) for record in salmonela_max_df["sequence"]]

# Compile motifs from alignments
num_clusters = 10
motifs = compile_motifs(num_clusters)
print(motifs)

# Use ProcessPoolExecutor for multithreading to match sequences against motifs
matched_sequences = set()
with ProcessPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(match_sequence_to_motifs, seq, motifs): seq for seq in sequences}
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        seq = futures[future]
        if future.result():
            matched_sequences.add(seq)



# If there are matching sequences, run a BLAST search
if matched_sequences:
    print("Matched Sequences:")
    for seq in matched_sequences:
        print(seq)

    # Save matched sequences to a FASTA file for BLAST
    with open(f"results/attention_visualized/{class_name}/Streptococcus_mutans_matched_sequences_{class_name}_1213_clus_10.fasta", "w") as f:
        for i, seq in enumerate(matched_sequences):
            f.write(f">seq{i + 1}\n{seq}\n")

    # # Run BLAST
    # blastp_cline = NcbiblastpCommandline(
    #     query=f"results/attention_visualized/{class_name}/matched_sequences_{class_name}_T0_8_1024_clus_10.fasta",
    #     db="datasets/Swissprot/swissprot_bacteria_db",  # or your local database name
    #     evalue=0.001,
    #     outfmt=5,
    #     out="blast_results.xml"
    # )

    # # Execute BLAST command
    # stdout, stderr = blastp_cline()
    # print("BLAST search completed. Results saved in blast_results.xml")
else:
    print("No sequences matched any of the motifs.")
    
fasta_sequences = SeqIO.parse(open(f"results/attention_visualized/{class_name}/Streptococcus_mutans_matched_sequences_{class_name}_1213_clus_10.fasta"), 'fasta')
sequences = [str(fasta.seq) for fasta in fasta_sequences]
matched_rows = []

for seq in sequences:
    seq = " ".join(seq)
    for idx, match in enumerate(salmonela_max_df["sequence"]):
        if match in seq:
            matched_rows.append(salmonela_max_df.iloc[idx])
            
matched_df = pd.DataFrame(matched_rows)
matched_df.to_csv(f"results/attention_visualized/{class_name}/Streptococcus_mutans_matched_sequences_{class_name}_1213_clus_10.csv", index=False)