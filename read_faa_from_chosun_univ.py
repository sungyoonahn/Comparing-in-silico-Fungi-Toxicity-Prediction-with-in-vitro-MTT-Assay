import pandas as pd
from Bio import SeqIO
import os
import re
import pandas as pd
import shutil
import gzip

pd.set_option("display.max_columns", None)

def faa2csv(faa_file):
    # faa_file = "datasets/bacteria_inference/chosun_univ_txt/GCF_000006945.2_ASM694v2_translated_cds.faa"

    data = []

    for record in SeqIO.parse(faa_file, "fasta"):
        description = record.description
        # Extract key-value pairs inside brackets
        pairs = re.findall(r"\[([^=]+)=([^\]]+)\]", description)
        # Convert to a dictionary
        record_data = {key: value for key, value in pairs}
        # Add the sequence ID and sequence to the dictionary
        record_data["ID"] = record.id
        record_data["sequence"] = str(record.seq)
        data.append(record_data)

    df = pd.DataFrame(data)
    df["sequence"] = df["sequence"].apply(lambda seq: " ".join(seq))

    df["label"] = [0]*len(df)
    index_no = []
    for i in range(len(df)):
        index_no.append(i)
    df["tensor id"] = index_no

    print(df)
    # df.to_csv("datasets/bacteria_inference/chosun_univ_txt/GCF_000006945.2_ASM694v2_translated_cds.csv", index=False)

    # df = pd.read_csv("datasets/bacteria_inference/chosun_univ_txt/LT2-genome-total.csv")
    # print(df)
    return df

print(os.listdir("sungyoon/env_project/datasets/bacteria_inference/NCBI_bacteria_zip_0218/"))
zip_root = "sungyoon/env_project/datasets/bacteria_inference/NCBI_bacteria_zip_0218/"
faa_root = "sungyoon/env_project/datasets/bacteria_inference/NCBI_bacteria_faa_0218/"
csv_root = "sungyoon/env_project/datasets/bacteria_inference/NCBI_bacteria_csv_0218/"
for tar_gz in os.listdir(zip_root):
    bacteria_name = tar_gz.split(".")[-3].split("_")[-1]
    original_filename = bacteria_name.replace(".gz", "")

    with gzip.open(zip_root+tar_gz, "rb") as f_in:
        with open(faa_root+original_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    os.rename(faa_root+original_filename, faa_root+bacteria_name+".faa")
    df = faa2csv(faa_root+bacteria_name+".faa")
    df.to_csv(csv_root+bacteria_name+".csv", index=False)


