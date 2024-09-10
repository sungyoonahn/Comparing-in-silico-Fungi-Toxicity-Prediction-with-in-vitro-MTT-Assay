from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio import AlignIO
import pandas as pd
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def compile_motifs(num_clusters):
    """
    Compile motifs from multiple sequence alignments.
    
    :param num_clusters: Number of clusters to process
    :return: List of compiled regex motifs
    """
    motifs = []
    for i in range(num_clusters):
        alignment = AlignIO.read(f"results/attention_visualized/{class_name}/cluster_{i + 1}_aligned_sequences.fasta", "fasta")
        most_likely_motif = find_most_likely_motif(alignment, threshold=0.8)
        
        first_index = next((i for i, char in enumerate(most_likely_motif) if char != '-'), None)
        last_index = next((i for i in range(len(most_likely_motif) - 1, -1, -1) if most_likely_motif[i] != '-'), None)
        
        trimmed_motif = most_likely_motif[first_index:last_index + 1]
        re_pattern = trimmed_motif.replace('-', '.*')  # Convert to regex pattern
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
salmonela_df = pd.read_csv("results/bacteria_inference/Salmonella_typhimurium_(strainLT2)0806.csv")
salmonela_anti_toxin_df = salmonela_df[salmonela_df["max label"] == "toxin"]
sequences = ["".join(record.split()) for record in salmonela_anti_toxin_df["sequence"]]

# Compile motifs from alignments
num_clusters = 5
motifs = compile_motifs(num_clusters)

# Use ProcessPoolExecutor for multithreading to match sequences against motifs
matched_sequences = set()
with ProcessPoolExecutor(max_workers=20) as executor:
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
    with open("test.fasta", "w") as f:
        for i, seq in enumerate(matched_sequences):
            f.write(f">seq{i + 1}\n{seq}\n")

    # Run BLAST
    blastp_cline = NcbiblastpCommandline(
        query="test.fasta",
        db="datasets/Swissprot/swissprot_bacteria_db",  # or your local database name
        evalue=0.001,
        outfmt=5,
        out="blast_results.xml"
    )

    # Execute BLAST command
    stdout, stderr = blastp_cline()
    print("BLAST search completed. Results saved in blast_results.xml")
else:
    print("No sequences matched any of the motifs.")
