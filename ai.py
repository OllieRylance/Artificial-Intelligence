import csv
import random
import numpy as np
import copy
import time

storeExtraData = True
frequencyOfExtraData = 200

# Function to calculate the sigmoid of a number
def sigmoid(x):
  # If x is greater than a large positive number, return a number close to 1.
  if x > 100:
    return 1.0
  # If x is less than a large negative number, return a number close to 0.
  elif x < -100:
    return 0.0
  else:
    return 1 / (1 + np.exp(-x))

# Function to calculate the derivative of the sigmoid of a number
def sigmoidDerivative(x):
  return sigmoid(x) * (1 - sigmoid(x))

# Function to calculate the tanh of a number
def tanh(x):
  # If x is greater than a large positive number, return a number close to 1.
  if x > 100:
    return 1.0
  # If x is less than a large negative number, return a number close to -1.
  elif x < -100:
    return -1.0
  else:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Function to calculate the derivative of the tanh of a number
def tanhDerivative(x):
  return 1 - tanh(x) ** 2

# Function to return the data
def returnData(splitDataDirectory, headingsInput):
  allHeadings = ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","Index flood"]
  # Read the three CSV files f'{splitDataDirectory}trainingData.txt', f'{splitDataDirectory}validationData.txt', and f'{splitDataDirectory}testingData.txt' and store them in lists called trainingData, validationData, and testingData
  trainingData = []
  with open(f'{splitDataDirectory}trainingData.txt', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      if row[0].startswith("#"):
        continue
      elementsToKeep = []
      for i in range(len(allHeadings)):
        if allHeadings[i] in headingsInput:
          elementsToKeep.append(row[i])
      trainingData.append(elementsToKeep)

  validationData = []
  with open(f'{splitDataDirectory}validationData.txt', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      elementsToKeep = []
      if row[0].startswith("#"):
        continue
      for i in range(len(allHeadings)):
        if allHeadings[i] in headingsInput:
          elementsToKeep.append(row[i])
      validationData.append(elementsToKeep)

  testingData = []
  with open(f'{splitDataDirectory}testingData.txt', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      if row[0].startswith("#"):
        continue
      elementsToKeep = []
      for i in range(len(allHeadings)):
        if allHeadings[i] in headingsInput:
          elementsToKeep.append(row[i])
      testingData.append(elementsToKeep)

  return trainingData, validationData, testingData

# Function to initialise the weights and biases
def initialiseWeightsBiases(numOfInputNodes, numOfHiddenNodes, numOfOutputNodes, weightsBiasesNumber): # !! need to make this function accept a list of hidden nodes from multiple layers
  # Initialise the weights, biases, and learning rate
  # Have a list of weights and biases for each layer
  # Set the fileName to the file that stores the weights and biases
  fileName = f'aiData/untrained/weightsBiases_{numOfInputNodes}Inputs{numOfHiddenNodes}Hiddens{numOfOutputNodes}Outputs_version{weightsBiasesNumber}.txt'
  try:
    # If a file exists with the weights and biases, read them from the file
    with open(fileName, 'r') as file:
      reader = csv.reader(file)

      # Create lists to store the weights and biases
      weightsFromInputToHidden = []
      weightsFromHiddenToOutput = []
      biasesFromInputToHidden = []
      biasesFromHiddenToOutput = []

      # Read the weights and biases from the file
      for row in reader:
        if row[0].startswith("#"):
          continue
        if len(weightsFromInputToHidden) < numOfInputNodes:
          weightsFromInputToHidden.append([float(i) for i in row])
        elif len(weightsFromHiddenToOutput) < numOfHiddenNodes:
          weightsFromHiddenToOutput.append([float(i) for i in row])
        elif len(biasesFromInputToHidden) < numOfHiddenNodes:
          biasesFromInputToHidden = [float(i) for i in row]
        elif len(biasesFromHiddenToOutput) < numOfOutputNodes:
          biasesFromHiddenToOutput = [float(i) for i in row]
  except FileNotFoundError:
    # If the file does not exist, create random weights and biases
    # The bounds for the random weights and biases are 2 / number of inputs for that node and -2 / number of inputs for that node
    # A 2D array for weights from input to hidden nodes, a 2D array for weights from hidden to output nodes, a 1D array for biases for the hidden nodes, and a 1D array for biases for the output nodes
    weightsFromInputToHidden = [[random.uniform(-2/numOfInputNodes, 2/numOfInputNodes) for j in range(numOfHiddenNodes)] for i in range(numOfInputNodes)]
    weightsFromHiddenToOutput = [[random.uniform(-2/numOfHiddenNodes, 2/numOfHiddenNodes) for j in range(numOfOutputNodes)] for i in range(numOfHiddenNodes)]
    biasesFromInputToHidden = [random.uniform(-2/numOfInputNodes, 2/numOfInputNodes) for i in range(numOfHiddenNodes)]
    biasesFromHiddenToOutput = [random.uniform(-2/numOfHiddenNodes, 2/numOfHiddenNodes) for i in range(numOfOutputNodes)]

    # Save the untrained weights and biases to a file
    saveWeightsBiasesUntrained(fileName, weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput)

  return weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput

# Function to save the trained weights and biases to a file
def saveWeightsBiasesUntrained(
    fileName, weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput
  ):
  # Create a new fileName to store the weights and biases before training
  with open(fileName, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["# Weights from input to hidden nodes:"])
    for row in weightsFromInputToHidden:
      writer.writerow(row)
    writer.writerow(["# Weights from hidden to output nodes:"])
    for row in weightsFromHiddenToOutput:
      writer.writerow(row)
    writer.writerow(["# Biases from input to hidden nodes:"])
    writer.writerow(biasesFromInputToHidden)
    writer.writerow(["# Biases from hidden to output nodes:"])
    writer.writerow(biasesFromHiddenToOutput)

# Function to save the trained weights and biases to a file
def saveWeightsBiasesTrained(
    fileName, weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput, epochs, testDataMeanError, numOfBackwardPassCalculations
):
  with open(fileName, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["# Weights from input to hidden nodes:"])
    for row in weightsFromInputToHidden:
      writer.writerow(row)
    writer.writerow(["# Weights from hidden to output nodes:"])
    for row in weightsFromHiddenToOutput:
      writer.writerow(row)
    writer.writerow(["# Biases from input to hidden nodes:"])
    writer.writerow(biasesFromInputToHidden)
    writer.writerow(["# Biases from hidden to output nodes:"])
    writer.writerow(biasesFromHiddenToOutput)
    writer.writerow(["# Number of epochs trained for:"])
    writer.writerow([epochs])
    writer.writerow(["# Mean squared error of finished training:"])
    writer.writerow([testDataMeanError])
    writer.writerow(["# Number of backward pass calculations:"])
    writer.writerow([numOfBackwardPassCalculations])

# Function to calculate the output of a layer during a forward pass
def forwardPass(input, currentWeights, currentBiases, numOfPreviousNodes, numOfCurrentNodes, activationFunction):
  # Calculate the outputs of layer using matrix multiplication
  # Convert the input into a matrix adding a bias to the start of the input
  npInput = np.array([1] + input)
  # Convert the weights and biases into matrices on the form [[b1  , b2  , b3  , ..., bn  ],
  #                                                           [w1 1, w1 2, w1 3, ..., w1 n],
  #                                                           [w2 1, w2 2, w2 3, ..., w2 4],
  #                                                           ...,
  #                                                           [wm 1, wm 2, wm 3, ..., wm n]] where n is the current node and m is the previous node
  biasesWeightsMatrix = [currentBiases]
  for i in range(numOfPreviousNodes):
    currentMatrixRow = [currentWeights[i][j] for j in range(numOfCurrentNodes)]
    biasesWeightsMatrix.append(currentMatrixRow)
  biasesWeightsMatrix = np.array(biasesWeightsMatrix)
  # Calculate the output of the layer
  currentOutput = np.matmul(npInput, biasesWeightsMatrix)
  # Apply the activation function to the every element of the output
  if activationFunction == "sigmoid":
    currentOutput = [sigmoid(i) for i in currentOutput]
  elif activationFunction == "tanh":
    currentOutput = [tanh(i) for i in currentOutput]
  elif activationFunction == "relu":
    currentOutput = [max(0, i) for i in currentOutput]

  return currentOutput

  # Calculate the outputs of layer without using matrix multiplication
  currentOutput = [0 for _ in range(numOfCurrentNodes)]
  for i in range(numOfCurrentNodes):
    currentOutput[i] = sum([input[j] * currentWeights[j][i] for j in range(numOfPreviousNodes)]) + currentBiases[i]
    if activationFunction == "sigmoid":
      currentOutput[i] = sigmoid(currentOutput[i])
    elif activationFunction == "tanh":
      currentOutput[i] = tanh(currentOutput[i])
    elif activationFunction == "relu":
      currentOutput[i] = max(0, currentOutput[i])
  
  return currentOutput

# Testing the weights and biases on the validation or testing data
def errorFunction(weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput, dataToTestWith, numOfInputNodes, numOfHiddenLayers, numOfHiddenNodes, numOfOutputNodes, activationFunctions):
  # Calculate the mean squared error of the testing data
  meanError = 0
  for row in dataToTestWith:
    # Get the input and output for the current row
    input = [float(i) for i in row[:-1]]
    output = float(row[-1])

    # Calculate the output of the hidden nodes
    hiddenOutput = [0 for i in range(numOfHiddenNodes)]
    for i in range(numOfHiddenNodes):
      hiddenOutput[i] = sum([input[j] * weightsFromInputToHidden[j][i] for j in range(numOfInputNodes)]) + biasesFromInputToHidden[i]
      if activationFunctions[0] == "sigmoid":
        hiddenOutput[i] = sigmoid(hiddenOutput[i])
      elif activationFunctions[0] == "tanh": 
        hiddenOutput[i] = tanh(hiddenOutput[i])
      elif activationFunctions[0] == "relu":
        hiddenOutput[i] = max(0, hiddenOutput[i])

    # Calculate the outputs of the output nodes
    finalOutputs = [0 for i in range(numOfOutputNodes)]
    for i in range(numOfOutputNodes):
      finalOutputs[i] = sum([hiddenOutput[j] * weightsFromHiddenToOutput[j][i] for j in range(numOfHiddenNodes)]) + biasesFromHiddenToOutput[i]
      if activationFunctions[numOfHiddenLayers] == "sigmoid":
        finalOutputs[i] = sigmoid(finalOutputs[i])
      elif activationFunctions[numOfHiddenLayers] == "tanh": 
        finalOutputs[i] = tanh(finalOutputs[i])
      elif activationFunctions[numOfHiddenLayers] == "relu":
        finalOutputs[i] = max(0, finalOutputs[i])

    # Add the squared error to the mean error
    meanError += (output - finalOutputs[0]) ** 2
  
  # Divide the mean error by the number of rows in the testing data
  meanError /= len(dataToTestWith)

  return meanError

# Function to train the neural network
def train(
    weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput, 
    trainingData, validationData, testingData,
    learningRate,
    numOfInputNodes, numOfHiddenLayers, numOfHiddenNodes, activationFunctions, numOfOutputNodes,
    maxEpochs, weightDecayInUse, momentumMultiplierInUse, momentumMultiplier, boldDriverInUse, boldDriverDecreaseFactor, boldDriverIncreaseFactor, frequencyOfBoldDriverChecks, maxErrorIncrease, maxLearningRate, minLearningRate, annealingInUse, annealingStartParmeter, annealingEndParmeter, frequencyOfValidationChecks, batchLearningInUse, numberOfBatchesSplitInto, fileName
):
  epochs = 0

  currentTime = time.time()

  if batchLearningInUse == True:
    listOfEndOfBatchNumbers = [int(np.ceil(i * (len(trainingData) / numberOfBatchesSplitInto))) for i in range(1, numberOfBatchesSplitInto + 1)]

  hiddenOutput = [0 for _ in range(numOfHiddenNodes)]
  finalOutputs = [0 for _ in range(numOfOutputNodes)]

  changesFromHiddenToOutputs = [0 for _ in range(numOfOutputNodes)]
  changesFromInputToHidden = [0 for _ in range(numOfHiddenNodes)]

  # Declare the number of backward pass calculations as a numpy int64
  numOfBackwardPassCalculations = np.int64(0)

  if storeExtraData == True:
    # Create a list to store the progression of the mean squared error of the testing data
    meanErrorProgression = []

  # Store variables to test changes of weights and biases as the network is trained
  """
  trackerOfAllForOneWeightForFirstFiftyRows = []
  """
  while epochs <= maxEpochs:

    # Store the sum of the changes in the weights and biases if batch learning is being used
    if batchLearningInUse == True:
      sumOfWeightChangesFromInputToHidden = [[0 for _ in range(numOfHiddenNodes)] for _ in range(numOfInputNodes)]
      sumOfWeightChangesFromHiddenToOutput = [[0 for _ in range(numOfOutputNodes)] for _ in range(numOfHiddenNodes)]
      sumOfBiasChangesFromInputToHidden = [0 for _ in range(numOfHiddenNodes)]
      sumOfBiasChangesFromHiddenToOutput = [0 for _ in range(numOfOutputNodes)]

      rows = 0
      recentRows = 0

    for row in trainingData:
      # Get the input and output for the current row
      input = [float(i) for i in row[:-1]]
      output = float(row[-1])

      # Forward Pass
      # Calculate the output of the hidden nodes and the output nodes
      """
      hiddenOutput = forwardPass(input, weightsFromInputToHidden, biasesFromInputToHidden, numOfInputNodes, numOfHiddenNodes, activationFunctions[0])
      """
      for i in range(numOfHiddenNodes):
        hiddenOutput[i] = sum([input[j] * weightsFromInputToHidden[j][i] for j in range(numOfInputNodes)]) + biasesFromInputToHidden[i]
        if activationFunctions[0] == "sigmoid":
          hiddenOutput[i] = sigmoid(hiddenOutput[i])
        elif activationFunctions[0] == "tanh":
          hiddenOutput[i] = tanh(hiddenOutput[i])
        elif activationFunctions[0] == "relu":
          hiddenOutput[i] = max(0, hiddenOutput[i])
      
      """
      finalOutputs = forwardPass(hiddenOutput, weightsFromHiddenToOutput, biasesFromHiddenToOutput, numOfHiddenNodes, numOfOutputNodes, activationFunctions[numOfHiddenLayers])
      """
      for i in range(numOfOutputNodes):
        finalOutputs[i] = sum([hiddenOutput[j] * weightsFromHiddenToOutput[j][i] for j in range(numOfHiddenNodes)]) + biasesFromHiddenToOutput[i]
        if activationFunctions[numOfHiddenLayers] == "sigmoid":
          finalOutputs[i] = sigmoid(finalOutputs[i])
        elif activationFunctions[numOfHiddenLayers] == "tanh":
          finalOutputs[i] = tanh(finalOutputs[i])
        elif activationFunctions[numOfHiddenLayers] == "relu":
          finalOutputs[i] = max(0, finalOutputs[i])
      

      if weightDecayInUse == False or epochs == 0:
        # Backward Pass
        # Calculate the changes in the output nodes
        for i in range(numOfOutputNodes):
          numOfBackwardPassCalculations += 1
          if activationFunctions[numOfHiddenLayers] == "sigmoid":
            changesFromHiddenToOutputs[i] = (output - finalOutputs[i]) * sigmoidDerivative(finalOutputs[i])
          elif activationFunctions[numOfHiddenLayers] == "tanh":
            changesFromHiddenToOutputs[i] = (output - finalOutputs[i]) * tanhDerivative(finalOutputs[i])
          elif activationFunctions[numOfHiddenLayers] == "relu":
            changesFromHiddenToOutputs[i] = (output - finalOutputs[i]) * (1 if finalOutputs[i] > 0 else 0)
      else:
        # Calculate the regularisation function omegas for the outputs
        regularisationFunctionOmegas = [0 for _ in range(numOfOutputNodes)]
        for i in range(numOfOutputNodes):
          for j in range(numOfHiddenNodes):
            regularisationFunctionOmegas[i] += weightsFromHiddenToOutput[j][i] ** 2
          regularisationFunctionOmegas[i] += biasesFromHiddenToOutput[i] ** 2
        regularisationFunctionOmegas = [regularisationFunctionOmegas[i] / (2 * (numOfHiddenNodes + 1)) for i in range(numOfOutputNodes)]

        # Update the regularisation parameter for weight decay
        regularisationParameter = 1 / (learningRate * epochs)

        # Backward Pass
        # Calculate the changes in the output nodes
        for i in range(numOfOutputNodes):
          numOfBackwardPassCalculations += 1
          if activationFunctions[numOfHiddenLayers] == "sigmoid":
            changesFromHiddenToOutputs[i] = (output - finalOutputs[i] + regularisationParameter * regularisationFunctionOmegas[i]) * sigmoidDerivative(finalOutputs[i])
          elif activationFunctions[numOfHiddenLayers] == "tanh":
            changesFromHiddenToOutputs[i] = (output - finalOutputs[i] + regularisationParameter * regularisationFunctionOmegas[i]) * tanhDerivative(finalOutputs[i])
          elif activationFunctions[numOfHiddenLayers] == "relu":
            changesFromHiddenToOutputs[i] = (output - finalOutputs[i] + regularisationParameter * regularisationFunctionOmegas[i]) * (1 if finalOutputs[i] > 0 else 0)

      # Calculate the change in the hidden nodes
      for i in range(numOfHiddenNodes):
        numOfBackwardPassCalculations += 1
        if activationFunctions[0] == "sigmoid":
          changesFromInputToHidden[i] = sum([weightsFromHiddenToOutput[i][j] * changesFromHiddenToOutputs[j] for j in range(numOfOutputNodes)]) * sigmoidDerivative(hiddenOutput[i])
        elif activationFunctions[0] == "tanh":
          changesFromInputToHidden[i] = sum([weightsFromHiddenToOutput[i][j] * changesFromHiddenToOutputs[j] for j in range(numOfOutputNodes)]) * tanhDerivative(hiddenOutput[i])
        elif activationFunctions[0] == "relu":
          changesFromInputToHidden[i] = sum([weightsFromHiddenToOutput[i][j] * changesFromHiddenToOutputs[j] for j in range(numOfOutputNodes)]) * (1 if hiddenOutput[i] > 0 else 0)


      # Store variables to test changes of weights and biases as the network is trained
      """
      if len(trackerOfAllForOneWeightForFirstFiftyRows) < 500:
        trackerOfAllForOneWeightForFirstFiftyRows.append([
          weightsFromInputToHidden[0][0] + learningRate * changesFromInputToHidden[0] * input[0],
          weightsFromInputToHidden[0][0] + learningRate * changesFromInputToHidden[0] * input[0] + momentumMultiplier * ((weightsFromInputToHidden[0][0] + learningRate * changesFromInputToHidden[0]) - weightsFromInputToHidden[0][0]),
          (weightsFromInputToHidden[0][0] * (1 - momentumMultiplier) + learningRate * changesFromInputToHidden[0] * input[0]) / (1 - momentumMultiplier)
        ])
      """

      # Update the weights and biases
      if batchLearningInUse == False:
        # If batch learning is not being used, update the weights and biases without batch learning
        if momentumMultiplierInUse == False:
          # If momentum is not being used, update the weights and biases without momentum
          for i in range(numOfHiddenNodes):
            for j in range(numOfInputNodes):
              weightsFromInputToHidden[j][i] += learningRate * changesFromInputToHidden[i] * input[j]
            for j in range(numOfOutputNodes):
              weightsFromHiddenToOutput[i][j] += learningRate * changesFromHiddenToOutputs[j] * hiddenOutput[i]
          for i in range(numOfHiddenNodes):
            biasesFromInputToHidden[i] += learningRate * changesFromInputToHidden[i]
          for i in range(numOfOutputNodes):
            biasesFromHiddenToOutput[i] += learningRate * changesFromHiddenToOutputs[i]
        else:
          # If momentum is being used, update the weights and biases using the momentum
          # This adapts the simple momentum formula: newWeight = oldWeight + learningRate * changes + momentumMultiplier * (oldWeight - oldOldWeight)
          # The momentumMultiplier is included in the standard weight and bias update formula
          for i in range(numOfHiddenNodes):
            for j in range(numOfInputNodes):
              weightsFromInputToHidden[j][i] += learningRate * (1 + momentumMultiplier) * changesFromInputToHidden[i] * input[j]
            for j in range(numOfOutputNodes):
              weightsFromHiddenToOutput[i][j] += learningRate * (1 + momentumMultiplier) * changesFromHiddenToOutputs[j] * hiddenOutput[i]
          for i in range(numOfHiddenNodes):
            biasesFromInputToHidden[i] += learningRate * (1 + momentumMultiplier) * changesFromInputToHidden[i]
          for i in range(numOfOutputNodes):
            biasesFromHiddenToOutput[i] += learningRate * (1 + momentumMultiplier) * changesFromHiddenToOutputs[i]
      else:
        # If batch learning is being used, update the weights and biases with batch learning
        rows += 1
        recentRows +=1

        # Update the sum of the changes in the weights and biases
        for i in range(numOfHiddenNodes):
          for j in range(numOfInputNodes):
            sumOfWeightChangesFromInputToHidden[j][i] += learningRate * changesFromInputToHidden[i] * input[j]
          for j in range(numOfOutputNodes):
            sumOfWeightChangesFromHiddenToOutput[i][j] += learningRate * changesFromHiddenToOutputs[j] * hiddenOutput[i]
        for i in range(numOfHiddenNodes):
          sumOfBiasChangesFromInputToHidden[i] += learningRate * changesFromInputToHidden[i]
        for i in range(numOfOutputNodes):
          sumOfBiasChangesFromHiddenToOutput[i] += learningRate * changesFromHiddenToOutputs[i]
        
        if momentumMultiplierInUse == False:
          # If momentum is not being used, update the sum of the changes in the weights and biases without momentum
          for i in range(numOfHiddenNodes):
            for j in range(numOfInputNodes):
              sumOfWeightChangesFromInputToHidden[j][i] += learningRate * changesFromInputToHidden[i] * input[j]
            for j in range(numOfOutputNodes):
              sumOfWeightChangesFromHiddenToOutput[i][j] += learningRate * changesFromHiddenToOutputs[j] * hiddenOutput[i]
          for i in range(numOfHiddenNodes):
            sumOfBiasChangesFromInputToHidden[i] += learningRate * changesFromInputToHidden[i]
          for i in range(numOfOutputNodes):
            sumOfBiasChangesFromHiddenToOutput[i] += learningRate * changesFromHiddenToOutputs[i]
        else:
          # If momentum is being used, update the sum of the changes in the weights and biases using the momentum
          for i in range(numOfHiddenNodes):
            for j in range(numOfInputNodes):
              sumOfWeightChangesFromInputToHidden[j][i] += learningRate * (1 + momentumMultiplier) * changesFromInputToHidden[i] * input[j]
            for j in range(numOfOutputNodes):
              sumOfWeightChangesFromHiddenToOutput[i][j] += learningRate * (1 + momentumMultiplier) * changesFromHiddenToOutputs[j] * hiddenOutput[i]
          for i in range(numOfHiddenNodes):
            sumOfBiasChangesFromInputToHidden[i] += learningRate * (1 + momentumMultiplier) * changesFromInputToHidden[i]
          for i in range(numOfOutputNodes):
            sumOfBiasChangesFromHiddenToOutput[i] += learningRate * (1 + momentumMultiplier) * changesFromHiddenToOutputs[i]

        # If the number of rows in the batch is correct, update the weights and biases and reset the sum of the changes in the weights and biases
        if rows in listOfEndOfBatchNumbers:
          for i in range(numOfHiddenNodes):
            for j in range(numOfInputNodes):
              weightsFromInputToHidden[j][i] += sumOfWeightChangesFromInputToHidden[j][i] / recentRows
            for j in range(numOfOutputNodes):
              weightsFromHiddenToOutput[i][j] += sumOfWeightChangesFromHiddenToOutput[i][j] / recentRows
          for i in range(numOfHiddenNodes):
            biasesFromInputToHidden[i] += sumOfBiasChangesFromInputToHidden[i] / recentRows
          for i in range(numOfOutputNodes):
            biasesFromHiddenToOutput[i] += sumOfBiasChangesFromHiddenToOutput[i] / recentRows
          sumOfWeightChangesFromInputToHidden = [[0 for _ in range(numOfHiddenNodes)] for _ in range(numOfInputNodes)]
          sumOfWeightChangesFromHiddenToOutput = [[0 for _ in range(numOfOutputNodes)] for _ in range(numOfHiddenNodes)]
          sumOfBiasChangesFromInputToHidden = [0 for _ in range(numOfHiddenNodes)]
          sumOfBiasChangesFromHiddenToOutput = [0 for _ in range(numOfOutputNodes)]
          recentRows = 0

    # Test the weights and biases on the validation data to make sure the model is not overfitting every 10th of the maxEpochs
    if boldDriverInUse == True and epochs % frequencyOfBoldDriverChecks == 0:
      currentBoldDriverMeanError = errorFunction(weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput, trainingData, numOfInputNodes, numOfHiddenLayers, numOfHiddenNodes, numOfOutputNodes, activationFunctions)
      # print(boldDriverMeanError)
      if epochs != 0:
        # If the mean error has increased by the predetermined amount, revert the weights and biases to the previous ones
        if currentBoldDriverMeanError > previousBoldDriverMeanError * (1 + maxErrorIncrease):
          weightsFromInputToHidden = previousWeightsFromInputToHiddenForBoldDriver
          weightsFromHiddenToOutput = previousWeightsFromHiddenToOutputForBoldDriver
          biasesFromInputToHidden = previousBiasesFromInputToHiddenForBoldDriver
          biasesFromHiddenToOutput = previousBiasesFromHiddenToOutputForBoldDriver
          if learningRate * boldDriverDecreaseFactor > minLearningRate:
            learningRate *= boldDriverDecreaseFactor
        elif currentBoldDriverMeanError < previousBoldDriverMeanError * (1 - maxErrorIncrease):
          if learningRate * boldDriverIncreaseFactor < maxLearningRate:
            learningRate *= boldDriverIncreaseFactor
      previousBoldDriverMeanError = currentBoldDriverMeanError
      # Store a copy of the weights and biases so they can be reverted to if the bold driver has caused the error function to increase by the predetermined amount
      previousWeightsFromInputToHiddenForBoldDriver = copy.deepcopy(weightsFromInputToHidden)
      previousWeightsFromHiddenToOutputForBoldDriver = copy.deepcopy(weightsFromHiddenToOutput)
      previousBiasesFromInputToHiddenForBoldDriver = copy.deepcopy(biasesFromInputToHidden)
      previousBiasesFromHiddenToOutputForBoldDriver = copy.deepcopy(biasesFromHiddenToOutput)
    elif annealingInUse == True:
      # Update the learning rate using the provided annealing formula
      learningRate = annealingEndParmeter + (annealingStartParmeter - annealingEndParmeter) * (1 - 1 / (1 + np.exp(10 - (20 * epochs / maxEpochs))))

    # If necessary, store the progression of the mean squared error of the testing data
    if storeExtraData == True and epochs % frequencyOfExtraData == 0:
      # Store the progression of the mean squared error of the testing data
      currentTestingMeanError = errorFunction(weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput, testingData, numOfInputNodes, numOfHiddenLayers, numOfHiddenNodes, numOfOutputNodes, activationFunctions)
      meanErrorProgression.append([epochs, currentTestingMeanError])

    # Test the weights and biases on the validation data to make sure the model is not overfitting every 10th of the maxEpochs
    if epochs % frequencyOfValidationChecks == 0:
      currentValidationMeanError = errorFunction(weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput, validationData, numOfInputNodes, numOfHiddenLayers, numOfHiddenNodes, numOfOutputNodes, activationFunctions)
      if __name__ == "__main__":
        if epochs != 0:
          print(f"current {currentValidationMeanError}, change {previousValidationMeanError - currentValidationMeanError:.20f}")
        else:
          print(f"current {currentValidationMeanError}")
        # Print the amount of time passes since the last time current time was set
        print(f"time {time.time() - currentTime}")
        currentTime = time.time()
      if epochs != 0:
        if currentValidationMeanError > previousValidationMeanError:
          weightsFromInputToHidden = previousWeightsFromInputToHiddenForValidation
          weightsFromHiddenToOutput = previousWeightsFromHiddenToOutputForValidation
          biasesFromInputToHidden = previousBiasesFromInputToHiddenForValidation
          biasesFromHiddenToOutput = previousBiasesFromHiddenToOutputForValidation
          break
      previousValidationMeanError = currentValidationMeanError

      # Store a copy of the weights and biases so they can be reverted to if the model is overfitting
      previousWeightsFromInputToHiddenForValidation = copy.deepcopy(weightsFromInputToHidden)
      previousWeightsFromHiddenToOutputForValidation = copy.deepcopy(weightsFromHiddenToOutput)
      previousBiasesFromInputToHiddenForValidation = copy.deepcopy(biasesFromInputToHidden)
      previousBiasesFromHiddenToOutputForValidation = copy.deepcopy(biasesFromHiddenToOutput)

    epochs += 1

  # Test changes of weights and biases as the network is trained
  """
  for i in trackerOfAllForOneWeightForFirstFiftyRows:
    # Print all elements of the tracker with equal spacing taking negative numbers into account
    # If a number is negative, it will take up one more space than a positive number
    toPrint = ""
    try:
      for j in i:
        if j < 0:
          toPrint += f"{j:.10f}   "
        else:
          toPrint += f" {j:.10f}   "
      print(toPrint)
    except TypeError:
      print(i)
  """

  # See how many epochs it took to train the model
  if __name__ == "__main__":
    print(f"epochs: {epochs}")
  if storeExtraData == True:
    # Save the list of the progression of the mean squared error of the testing data to a file in the format of:
    with open(f"aiData/extraData/meanErrorProgression_{fileName[29:]}.", "w") as file:
      for dataPoint in meanErrorProgression:
        file.write(f"{dataPoint[0]}\t{dataPoint[1]}\n")

  return epochs, numOfBackwardPassCalculations

# Main function with many default parameters for the user to change
def main(  
  # Create a list of headings for the data
  headings = ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","Index flood"],

  # Set the number of hidden layers
  numOfHiddenLayers = 1,

  # Set the number of hidden and output nodes in the layers
  numOfHiddenNodesInEachLayer = [12],
  numOfOutputNodes = 1,

  # Set the activation function for each layer
  activationFunctions = ["sigmoid", "sigmoid"],

  # Setting learning parameter
  learningRate = 0.1,

  # Set the maximum number of epochs and the frequency of validation checks
  maxEpochs = 10000,
  frequencyOfValidationChecks = 500,

  # Set the number of the weights and biases file
  weightsBiasesNumber = 0,

  # Set the directory of the split data
  randomSplitNumber = -1,

  # Set the stats about momentum
  momentumMultiplierInUse = False,
  momentumMultiplier = 0.9,

  # Set if either bold driver or annealing is in use
  eitherBoldDriverOrAnnealingInUse = False,

  # Set bold driver parameters
  boldDriverInUse = False,
  boldDriverIncreaseFactor = 1.05,
  boldDriverDecreaseFactor = 0.7,
  frequencyOfBoldDriverChecks = 10,
  maxErrorIncrease = 0.04,

  maxLearningRate = 0.5,
  minLearningRate = 0.01,

  # Set annealing parameters
  annealingStartParmeter = 0.1,
  annealingEndParmeter = 0.01,

  # Set weight decay parameters
  weightDecayInUse = False,

  # Set batch learning parameters
  batchLearningInUse = False,
  numberOfBatchesSplitInto = 20
):
  # Set the number of inputs based on the input headings
  numOfInputNodes = len(headings) - 1

  # Update the bold driver in use variable to be False if either bold driver or annealing is in use is False
  if eitherBoldDriverOrAnnealingInUse == False:
    boldDriverInUse = False

  # Set whether annealing is in use based on if either bold driver or annealing is in use and if bold driver is not in use
  # To stop both bold driver and annealing from being used at the same time
  if eitherBoldDriverOrAnnealingInUse == True and boldDriverInUse == False:
    annealingInUse = True
  else:
    annealingInUse = False

  # Set the directory of the split data
  if randomSplitNumber == -1:
    splitDataDirectory = "splitData/notRandom/"
  else:
    splitDataDirectory = f"splitData/random/dataSet{randomSplitNumber}/"

  # Create a new fileName to store the weights and biases after training in the form of:
  # weightsBiases_
  fileName = f'aiData/trained/weightsBiases_'
  # *binary number of with 1 representing that input is used and 0 representing that it is not (in order), i.e., 1111111 means all inputs are used, 1100000 means only AREA and BFIHOST are inputs*;
  for input in ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","Index flood"]:
    if input in headings:
      fileName += "1"
    else:
      fileName += "0"
  fileName += ","
  # *number of hidden layers, i.e., 1*,
  fileName += str(numOfHiddenLayers) + ","
  # (*number of hidden nodes in hidden layer 1, i.e., 8*, *(if exists) number of hidden nodes in hidden layer n, i.e., 6*),
  fileName += "("
  for index, numOfHiddenNodes in enumerate(numOfHiddenNodesInEachLayer):
    if index == 0:
      fileName += str(numOfHiddenNodes)
    else:
      fileName += f",{numOfHiddenNodes}"
  fileName += "),"
  # (*activation function for hidden layer 1, i.e., sigmoid or tanh*, *(if exists) activation function for hidden layer n, i.e., sigmoid or tanh, activation function for output layer n, i.e., sigmoid or tanh*),
  fileName += "("
  for index, activationFunction in enumerate(activationFunctions):
    if index == 0:
      fileName += activationFunction
    else:
      fileName += f",{activationFunction}"
  fileName += "),"
  # *number of output nodes, i.e., 1*,
  fileName += f"{numOfOutputNodes},"
  # *initial learning rate*, 
  fileName += f"{learningRate},"
  # *maximum number of epochs*,
  fileName += f"{maxEpochs},"
  # *frequency of validation checks*,
  fileName += f"{frequencyOfValidationChecks},"
  # *weight biases file number*,
  fileName += f"{weightsBiasesNumber},"
  # *random split folder number*,
  fileName += f"{randomSplitNumber},"
  # *momentum multiplier in use, i.e., 0 or 1*,
  fileName += f"{int(momentumMultiplierInUse)},"
  # *momentum multiplier value*,
  fileName += f"{momentumMultiplier},"  
  # *either bold driver or annealing in use, i.e., 0 or 1*,
  fileName += f"{int(eitherBoldDriverOrAnnealingInUse)},"
  # *bold driver in use, i.e., 0 or 1*,
  fileName += f"{int(boldDriverInUse)},"
  # *bold driver increase factor*,
  fileName += f"{boldDriverIncreaseFactor},"
  # *bold driver decrease factor*,
  fileName += f"{boldDriverDecreaseFactor},"
  # *frequency of bold driver checks*,
  fileName += f"{frequencyOfBoldDriverChecks},"
  # *maximum error increase*,
  fileName += f"{maxErrorIncrease},"
  # *maximum learning rate*,
  fileName += f"{maxLearningRate},"
  # *minimum learning rate*,
  fileName += f"{minLearningRate},"
  # *annealing start parameter*,
  fileName += f"{annealingStartParmeter},"
  # *annealing end parameter*,
  fileName += f"{annealingEndParmeter},"
  # *weight decay in use, i.e., 0 or 1*,
  fileName += f"{int(weightDecayInUse)},"
  # *batch learning in use, i.e., 0 or 1*,
  fileName += f"{int(batchLearningInUse)},"
  # *number of batches split into*
  fileName += f"{numberOfBatchesSplitInto}"

  fileName += '.txt'

  # Run the main code of the main function if the parameters are new
  try:
    with open(fileName, 'r') as _:
      pass
    if __name__ == "__main__":
      print(f"The weights and biases have already been trained with these parameters and sorted in {fileName}. If you want to train the weights and biases again, change the parameters at the top of the file.")
    return None, None
  except FileNotFoundError:
    # Function to load the data
    trainingData, validationData, testingData = returnData(splitDataDirectory, headings)

    # Function to initialise the weights and biases
    weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput = initialiseWeightsBiases(numOfInputNodes, numOfHiddenNodesInEachLayer[0], numOfOutputNodes, weightsBiasesNumber)

    epochs, numOfBackwardPassCalculations = train(
      weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput,
      trainingData, validationData, testingData,
      learningRate,
      numOfInputNodes, numOfHiddenLayers, numOfHiddenNodesInEachLayer[0], activationFunctions, numOfOutputNodes,
      maxEpochs, weightDecayInUse, momentumMultiplierInUse, momentumMultiplier, boldDriverInUse, boldDriverDecreaseFactor, boldDriverIncreaseFactor, frequencyOfBoldDriverChecks, maxErrorIncrease, maxLearningRate, minLearningRate, annealingInUse, annealingStartParmeter, annealingEndParmeter, frequencyOfValidationChecks, batchLearningInUse, numberOfBatchesSplitInto, fileName
    )

    testDataMeanError = errorFunction(weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput, testingData, numOfInputNodes, numOfHiddenLayers, numOfHiddenNodesInEachLayer[0], numOfOutputNodes, activationFunctions)

    # Print the mean error of the testing data
    if __name__ == "__main__":
      print(testDataMeanError)

    # Save the trained weights and biases to a file
    saveWeightsBiasesTrained(
      fileName, weightsFromInputToHidden, weightsFromHiddenToOutput, biasesFromInputToHidden, biasesFromHiddenToOutput, epochs, testDataMeanError, numOfBackwardPassCalculations
    )

    # Print that the weights and biases have been trained and saved to a file
    if __name__ == "__main__":
      print(f"The weights and biases have been trained and saved to the file '{fileName}'.")

    return testDataMeanError, epochs
  
if __name__ == "__main__":
  # firstTime = time.time()
  main(maxEpochs=10001, weightsBiasesNumber=2, randomSplitNumber=2)
  # print(f"Time taken: {time.time() - firstTime}")
