import csv
import random
import os

def main(randomSplitNum = None):
  # Read the CSV file "unsplitData/processedData.txt" and store it in a list called dataPredictors
  dataPredictors = []
  with open('unsplitData/processedData.txt', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      if row[0].startswith("#"):
        continue
      dataPredictors.append(row)

  # Split the dataPredictors list into three lists: one for training, one for validation, and one for testing
  # Make the split 60% training, 20% validation, and 20% testing
  if randomSplitNum == None:
    trainingData = dataPredictors[:int(len(dataPredictors)*0.6)]
    validationData = dataPredictors[int(len(dataPredictors)*0.6):int(len(dataPredictors)*0.8)]
    testingData = dataPredictors[int(len(dataPredictors)*0.8):]
  else:
    # Randomly shuffle the dataPredictors list
    random.shuffle(dataPredictors)
    trainingData = dataPredictors[:int(len(dataPredictors)*0.6)]
    validationData = dataPredictors[int(len(dataPredictors)*0.6):int(len(dataPredictors)*0.8)]
    testingData = dataPredictors[int(len(dataPredictors)*0.8):]

  # Standardise the data so that the largest value in each column is 0.9 and the smallest is 0.1
  # Take the maximum and minimum values of each column in training data and validation data and then use these to standardise all three lists

  # Create a list of headings for the data
  headings = ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","Index flood"]

  for columnIndex in range(len(headings)):
    # Combine each column of trainingData and validationData to find the minimum and maximum element of them
    column1 = [float(row[columnIndex]) for row in trainingData]
    column2 = [float(row[columnIndex]) for row in validationData]
    combinedColumn = column1 + column2
    minElement = min(combinedColumn)
    maxElement = max(combinedColumn)
    difference = maxElement - minElement
    # To standardise between lowerBound and 1 - lowerBound, use the formula (x - min) / (max - min) * (1 - 2 * lowerBound) + lowerBound 
    # If the lowerbound were 0.1, use the formula (x - min) / (max - min) * 0.8 + 0.1 on each list
    lowerBound = 0.1
    boundRange = 1 - 2 * lowerBound
    for row in trainingData:
      row[columnIndex] = (float(row[columnIndex]) - minElement) / difference * boundRange + lowerBound
    for row in validationData:
      row[columnIndex] = (float(row[columnIndex]) - minElement) / difference * boundRange + lowerBound
    for row in testingData:
      row[columnIndex] = (float(row[columnIndex]) - minElement) / difference * boundRange + lowerBound

  # Test to make sure the data was split correctly
  """
  print(trainingData[-1])
  print(validationData[0])
  print(validationData[-1])
  print(testingData[0])
  """

  if randomSplitNum == None:
    # Write the three lists to three separate files
    with open('splitData/notRandom/trainingData.txt', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(trainingData)
    with open('splitData/notRandom/validationData.txt', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(validationData)
    with open('splitData/notRandom/testingData.txt', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(testingData)
  else:
    # Write the three lists to three separate files in the folder specified by randomSplitNum
    os.makedirs(f'splitData/random/dataSet{randomSplitNum}')
    with open(f'splitData/random/dataSet{randomSplitNum}/trainingData.txt', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(trainingData)
    with open(f'splitData/random/dataSet{randomSplitNum}/validationData.txt', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(validationData)
    with open(f'splitData/random/dataSet{randomSplitNum}/testingData.txt', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(testingData)

if __name__ == "__main__":
  main()