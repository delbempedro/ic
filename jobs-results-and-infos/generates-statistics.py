#open file and save witch contains in "contains"
file_name = 'job-cv6cvw7w1n80008zp270-result.json'
file_name = 'job-cv6cvw7w1n80008zp270/'+file_name
file = open(file_name, 'r', encoding='utf-8')
contains = file.readlines()

#create final values list and append values finded in "contains"
final_values = []
for caracters in contains:
    position = caracters.find("x")
    if position != -1:
        caracters = caracters[position+1]
        final_values.append(caracters)

#close file
file.close()

#create measures list and statistics dictionary
measures = [""]*1024
statistics = {}

#generate the measures getting the values in "final_values" every 1024(number of measures) bits
for start in range(1024):
    for value in range(start, len(final_values), 1024):
        measures[start] = measures[start] + final_values[value]
    if measures[start] in statistics:
        statistics[measures[start]] += 1
    else:
        statistics[measures[start]] = 1

print(statistics)