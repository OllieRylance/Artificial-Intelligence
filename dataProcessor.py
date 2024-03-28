import csv
import numpy as np
import matplotlib.pyplot as plt

# Open the text document called "rawData.py"
file_name = "unsplitData/rawData.txt"

# Create a list of headings for the data
headings = ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","Index flood"]

# In case file does not work
data_lines = []

try:
  with open(file_name, 'r') as file:
    # Create a list of each line of data stored as a string
    data_lines = [line.strip() for line in file.readlines()]
    
    # Split each line of code by tabs and store them as a list in a list called "dataPredictors"
    dataPredictors = [line.split("\t") for line in data_lines]
except FileNotFoundError:
  print(f"File '{file_name}' not found.")

# Remove all rows from dataPredictors that have negative data
# Create a new list to store the rows that have no negative data or impossible data
possibleDataPredictors = []

# Create a list to store the rows that have negative data
eroniousRows = []

# Create a list to store the rows that have impossible data
impossibleValues = []

# Go through each row in the dataPredictors list
for rowIndex, row in enumerate(dataPredictors):
  floatRow = []
  
  # Assume that the row has no negative data
  has_issue = False
  
  # Go through each element in the row
  try:
    for elementIndex, element in enumerate(row):
      floatRow.append(float(element))
  except ValueError:
    # If the element is not a number, set has_issue to True
    eroniousRows.append([[rowIndex, elementIndex], row])
    has_issue = True
  
  if has_issue == False:
    for elementIndex, element in enumerate(row):
      if floatRow[elementIndex] < 0:
        impossibleValues.append([[rowIndex, elementIndex], floatRow])
        has_issue = True
        break

  PROPWETIndex = headings.index("PROPWET")
  if has_issue == False and floatRow[PROPWETIndex] > 1:
    impossibleValues.append([[rowIndex, PROPWETIndex], floatRow])
    has_issue = True

  # If the row has no issues, add it to the new list
  if not has_issue:
    possibleDataPredictors.append(floatRow)


# Normalise the data
for row in possibleDataPredictors:
  # Apply log 10 to the columns that share an index with the headings "AREA", "FPEXT", "LDP", "SAAR", and "Index flood"
  row[headings.index("AREA")] = np.log10(row[headings.index("AREA")])
  row[headings.index("FPEXT")] = np.log10(row[headings.index("FPEXT")])
  row[headings.index("LDP")] = np.log10(row[headings.index("LDP")])
  row[headings.index("SAAR")] = np.log10(row[headings.index("SAAR")])
  row[headings.index("Index flood")] = np.log10(row[headings.index("Index flood")])

  # Apply the formula x = (x ** lambdaValue - 1) / lambdaValue to the column that shares a heading with "FARL"
  lambdaValue = 11
  row[headings.index("FARL")] = (row[headings.index("FARL")] ** lambdaValue - 1) / lambdaValue

# Outlier Removing using Mean and Standard Deviation Calculation
# Go through each line in the dataPredictors list and find the mean and standard deviation each column
# Convert the list to a numpy array for easier calculations
data_array = np.array(possibleDataPredictors, dtype=float)

# Calculate the mean and standard deviation of each column
means = np.mean(data_array, axis=0)
std_devs = np.std(data_array, axis=0)

outliers = []
outlierlessData = []
columnsWithOutliers = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0}


# Go through each row of the possibleDataPredictors list
for rowIndex, row in enumerate(possibleDataPredictors):
  # Go through each element in the row
  for elementIndex, element in enumerate(row):
    contains_outlier = False
    # Add the row's which contain values that are more than the max standard deviations away from the mean to the outliers list
    maxStandardDeviations = 5
    if element > means[elementIndex] + maxStandardDeviations * std_devs[elementIndex] or element < means[elementIndex] - maxStandardDeviations * std_devs[elementIndex]:
      outliers.append([[rowIndex, elementIndex], row])
      columnsWithOutliers[str(elementIndex)] += 1
      contains_outlier = True
      break

  if contains_outlier == False:
    outlierlessData.append(row)

# Plotting
# Create a 3x3 grid of bar charts to show the distribution of the data
fig, axs = plt.subplots(3, 3)

# Give the figure "fig" with the title "Data Distribution"
fig.suptitle("Data Distribution")

for headingIndex in range(len(headings)):
  # Plot the values from index 0 of each row on a bar chart
  elementList = [row[headingIndex] for row in outlierlessData] # Needs to change to outlierlessData
  elementList.sort()

  minElement = elementList[0]
  maxElement = elementList[-1]

  difference = maxElement - minElement
  iterations = 15

  numberOfValuesInEachIteration = []

  for i in range(iterations):
    numberOfValuesInEachIteration.append(0)

  for row in outlierlessData: # Needs to change to outlierlessData
    currentElement = row[headingIndex]
    index = int((currentElement - minElement) / (difference / iterations))
    if index == iterations:
      index -= 1
    try:
      numberOfValuesInEachIteration[index] += 1
    except IndexError:
      print(currentElement, minElement, maxElement, difference, index)      

  axs[headingIndex // 3, headingIndex % 3].bar(range(iterations), numberOfValuesInEachIteration)
  axs[headingIndex // 3, headingIndex % 3].set_title(headings[headingIndex])

# Show the bar chart
plt.tight_layout()
plt.show()

# Print the eronious rows, impossible values, and outliers
print("Eronious Rows:")
if len(eroniousRows) == 0:
  print("None")
else:
  for i in eroniousRows:
    print(i)

print("\nImpossible Rows:")
if len(impossibleValues) == 0:
  print("None")
else:
  for i in impossibleValues:
    print(i)
  
print("\nOutliers:")
if len(outliers) == 0:
  print("None")
else:
  for i in outliers:
    print(i)

print("\nOutlier Info:")
for i in columnsWithOutliers:
  print(f"After normalisation, {headings[int(i)]} had {columnsWithOutliers[i]} outliers within {maxStandardDeviations} standard deviations of the mean.")

print("\nPercentage of data left:")
print(len(outlierlessData)/len(dataPredictors))

# Store the processed data
output_file_name = "unsplitData/processedData.txt"

# Write the processed data to a new CSV file
with open(output_file_name, 'w', newline='') as file:
  writer = csv.writer(file)
  writer.writerows(outlierlessData)