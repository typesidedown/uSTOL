
import csv

def returnCamberLine(camberLineFileLocation):
    with open(camberLineFileLocation, mode='r', encoding='utf-8') as file:
        # Create a csv.reader object to iterate over lines in the CSV file
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        xCamber = []
        yCamber = []

        x_index = header.index('X')
        y_index = header.index('Y')

        for row in csv_reader:
            xCamber.append(float(row[x_index]))
            yCamber.append(float(row[y_index]))
    
    chord = max(xCamber) - min(xCamber)
    xCamber = [x / chord for x in xCamber]
    yCamber = [y / chord for y in yCamber]
    
    return(xCamber, yCamber)
