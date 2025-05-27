from tqdm import tqdm
import csv

with open(r"4way_ortholog.alignments", "r") as f:
    all_lines = f.readlines()

output_csv = r"4way_ortholog_by_row.csv"

with open(output_csv, "w") as csvfile:
    writer = csv.writer(csvfile)

    current_dna_string = ""
    current_csv_line = []

    for line in tqdm(all_lines):
        #print(line)
        # either the line starts with a '>' meaning new paragraph
        # or starts with dna meaning part of existing paragraph
        if line[0] == "=" or line[0] == ">":
            #add the dna string to the csv line and add it to the file
            current_csv_line.append(current_dna_string)
            current_dna_string = ""
            writer.writerow(current_csv_line)
                
            current_csv_line = []
            current_csv_line.append(line.split('\n')[0])        
        else:
            # add the line without the '\n' at the end
            current_dna_string+=line.split('\n')[0]




    

