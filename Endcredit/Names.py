import csv

def read_names_from_file(file_path, full_name=False):
    with open(file_path, 'r') as file:
        if full_name:
            names = [line.strip().lower() for line in file.readlines() if line.strip()]
        else:
            names = set(line.strip().lower() for line in file.readlines() if line.strip())
    return names

def compare_names(file1_names, file2_names):
    common_names = set()
    for name in file2_names:
        parts = name.split()
        for part in parts:
            if part.lower() in file1_names:
                common_names.add(name)  
                break  
    return common_names

def write_common_names_to_csv(common_names, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Common Names"])  
        for name in sorted(common_names):  
            writer.writerow([name])  

file1_path = '/Users/srragulraj/Desktop/end/names.csv'  
file2_path = '/Users/srragulraj/Desktop/end/extracted_names.txt'  
output_csv_path = 'common_names.csv'

names_file1 = read_names_from_file(file1_path, full_name=False)  
names_file2 = read_names_from_file(file2_path, full_name=True)  

common_names = compare_names(names_file1, names_file2)

if common_names:
    write_common_names_to_csv(common_names, output_csv_path)
    print(f"Common names have been written to {output_csv_path}")
else:
    print("No common names found.")
