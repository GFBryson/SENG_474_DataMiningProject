import csv

csv_file = open ('./datasets/government_response_data.csv', 'r')

reader = csv.reader(csv_file, delimiter=',')


index = 0
Wk4_column = 0
Wk3_column = 0
Wk2_column = 0
Wk1_column = 0
csv_rows = []
for row in reader:
    if index == 0:
        csv_rows.append(",".join(row))
        index_column = row.index("HealthIndexChange")
        Wk4_column = row.index("Cases after 4 weeks")
        Wk3_column = row.index("Cases after 3 weeks")
        Wk2_column = row.index("Cases after 2 weeks")
        Wk1_column = row.index("Cases after 1 week")
        index+=1
        continue
    index+=1
    if int(row[index_column]) > 0 and row[Wk4_column] is not '' and row[Wk3_column] is not '' and row[Wk2_column] is not '' and row[Wk1_column] is not '':
        csv_rows.append(",".join(row))

csv = """
""".join(csv_rows)

out_file = open('./datasets/government_response_data_cleaned.csv', 'w')
out_file.write(csv)
out_file.close()