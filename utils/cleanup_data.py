import csv

csv_file = open ('./datasets/government_response_data.csv', 'r')

reader = csv.reader(csv_file, delimiter=',')


index = 0
Wk4_column = 0
Wk3_column = 0
Wk2_column = 0
Wk1_column = 0
csv_rows_1 = []
csv_rows_2 = []
csv_rows_3 = []
csv_rows_4 = []
for row in reader:
    if index == 0:
        # csv_rows.append(",".join(row))
        index_column = row.index("HealthIndexChange")
        Wk4_column = row.index("Cases after 4 weeks")
        Wk3_column = row.index("Cases after 3 weeks")
        Wk2_column = row.index("Cases after 2 weeks")
        Wk1_column = row.index("Cases after 1 week")
        index+=1
        continue
    index+=1
    if int(row[index_column]) > 0 and row[Wk4_column] is not '' :
        csv_rows_4.append(",".join(row))
    if int(row[index_column]) > 0 and row[Wk3_column] is not '' :
        csv_rows_3.append(",".join(row))
    if int(row[index_column]) > 0 and row[Wk4_column] is not '' :
        csv_rows_2.append(",".join(row))
    if int(row[index_column]) > 0 and row[Wk1_column] is not '' :
        csv_rows_1.append(",".join(row))


csv_1 = """
""".join(csv_rows_1)
out_file = open('./datasets/government_response_data_cleaned_WK1.csv', 'w')
out_file.write(csv_1)

csv_2 = """
""".join(csv_rows_2)
out_file = open('./datasets/government_response_data_cleaned_WK2.csv', 'w')
out_file.write(csv_2)

csv_3 = """
""".join(csv_rows_3)
out_file = open('./datasets/government_response_data_cleaned_WK3.csv', 'w')
out_file.write(csv_3)

csv_4 = """
""".join(csv_rows_4)
out_file = open('./datasets/government_response_data_cleaned_WK4.csv', 'w')
out_file.write(csv_4)

out_file.close()