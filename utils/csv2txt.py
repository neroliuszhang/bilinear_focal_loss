import csv

csv_file="/media/goerlab/My Passport/Welder_detection/dataset/20180206/middle_data/val.csv"
csv_reader=csv.reader(open(csv_file))
total=csv_reader.line_num

txt_name="/media/goerlab/My Passport/Welder_detection/dataset/20180206/middle_data/val.txt"
generate_train_txt=open(txt_name,"wb")
generate_train_txt_file=file(txt_name,"r+")
for row in csv_reader:
    file_name=row[0]
    label=row[1]
    new_line=file_name+" "+label+'\n'
    generate_train_txt_file.write(new_line)