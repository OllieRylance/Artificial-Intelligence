import csv
from ai import returnData, sigmoid, tanh

justOptimal = True

def returnFileName(headings, numOfHiddenLayers, numOfHiddenNodesInEachLayer, activationFunctions, numOfOutputNodes, learningRate, maxEpochs, frequencyOfValidationChecks, weightsBiasesNumber, randomSplitNumber, momentumMultiplierInUse, momentumMultiplier, eitherBoldDriverOrAnnealingInUse, boldDriverInUse, boldDriverIncreaseFactor, boldDriverDecreaseFactor, frequencyOfBoldDriverChecks, maxErrorIncrease, maxLearningRate, minLearningRate, annealingStartParmeter, annealingEndParmeter, weightDecayInUse, batchLearningInUse, numberOfBatchesSplitInto):
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

  return fileName

def returnFileNames(index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList):
  if justOptimal == False:
    # fileNames in the form of:
    # {
    #   str(*option1*): [*fileNames*]
    # }
    fileNames = {}
    for i in range(len(weightsBiasesNumberOptionList)):
      for j in range(len(randomSplitNumberOptionList)):
        if index == 0:
          for k in range(len(headingsOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(headingsOptionList[k]) not in fileNames:
              fileNames[str(headingsOptionList[k])] = []
            fileNames[str(headingsOptionList[k])].append(returnFileName(headingsOptionList[k], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 1:
          for k in range(len(numOfHiddenLayersOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(numOfHiddenLayersOptionList[k]) not in fileNames:
              fileNames[str(numOfHiddenLayersOptionList[k])] = []
            fileNames[str(numOfHiddenLayersOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[k], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 2:
          for k in range(len(numOfHiddenNodesInEachLayerOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(numOfHiddenNodesInEachLayerOptionList[k]) not in fileNames:
              fileNames[str(numOfHiddenNodesInEachLayerOptionList[k])] = []
            fileNames[str(numOfHiddenNodesInEachLayerOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[k], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 3:
          for k in range(len(activationFunctionsOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(activationFunctionsOptionList[k]) not in fileNames:
              fileNames[str(activationFunctionsOptionList[k])] = []
            fileNames[str(activationFunctionsOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[k], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 4:
          for k in range(len(numOfOutputNodesOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(numOfOutputNodesOptionList[k]) not in fileNames:
              fileNames[str(numOfOutputNodesOptionList[k])] = []
            fileNames[str(numOfOutputNodesOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[k], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 5:
          for k in range(len(learningRateOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(learningRateOptionList[k]) not in fileNames:
              fileNames[str(learningRateOptionList[k])] = []
            fileNames[str(learningRateOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[k], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 6:
          for k in range(len(maxEpochsOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(maxEpochsOptionList[k]) not in fileNames:
              fileNames[str(maxEpochsOptionList[k])] = []
            fileNames[str(maxEpochsOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[k], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 7:
          for k in range(len(frequencyOfValidationChecksOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(frequencyOfValidationChecksOptionList[k]) not in fileNames:
              fileNames[str(frequencyOfValidationChecksOptionList[k])] = []
            fileNames[str(frequencyOfValidationChecksOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[k], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 8:
          for k in range(len(weightsBiasesNumberOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(weightsBiasesNumberOptionList[k]) not in fileNames:
              fileNames[str(weightsBiasesNumberOptionList[k])] = []
            fileNames[str(weightsBiasesNumberOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[k], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 9:
          for k in range(len(randomSplitNumberOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(randomSplitNumberOptionList[k]) not in fileNames:
              fileNames[str(randomSplitNumberOptionList[k])] = []
            fileNames[str(randomSplitNumberOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[k], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 10:
          for k in range(len(momentumMultiplierInUseOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(momentumMultiplierInUseOptionList[k]) not in fileNames:
              fileNames[str(momentumMultiplierInUseOptionList[k])] = []
            fileNames[str(momentumMultiplierInUseOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[k], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 11:
          for k in range(len(momentumMultiplierOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(momentumMultiplierOptionList[k]) not in fileNames:
              fileNames[str(momentumMultiplierOptionList[k])] = []
            fileNames[str(momentumMultiplierOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[k], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 12:
          for k in range(len(eitherBoldDriverOrAnnealingInUseOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(eitherBoldDriverOrAnnealingInUseOptionList[k]) not in fileNames:
              fileNames[str(eitherBoldDriverOrAnnealingInUseOptionList[k])] = []
            fileNames[str(eitherBoldDriverOrAnnealingInUseOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[k], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 13:
          for k in range(len(boldDriverInUseOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(boldDriverInUseOptionList[k]) not in fileNames:
              fileNames[str(boldDriverInUseOptionList[k])] = []
            if boldDriverInUseOptionList[k] == False:
              fileNames[str(boldDriverInUseOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[k], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
            else:
              fileNames[str(boldDriverInUseOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[1], boldDriverInUseOptionList[k], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 14:
          for k in range(len(boldDriverIncreaseFactorOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(boldDriverIncreaseFactorOptionList[k]) not in fileNames:
              fileNames[str(boldDriverIncreaseFactorOptionList[k])] = []
            fileNames[str(boldDriverIncreaseFactorOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[k], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 15:
          for k in range(len(boldDriverDecreaseFactorOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(boldDriverDecreaseFactorOptionList[k]) not in fileNames:
              fileNames[str(boldDriverDecreaseFactorOptionList[k])] = []
            fileNames[str(boldDriverDecreaseFactorOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[k], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 16:
          for k in range(len(frequencyOfBoldDriverChecksOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(frequencyOfBoldDriverChecksOptionList[k]) not in fileNames:
              fileNames[str(frequencyOfBoldDriverChecksOptionList[k])] = []
            fileNames[str(frequencyOfBoldDriverChecksOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[k], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 17:
          for k in range(len(maxErrorIncreaseOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(maxErrorIncreaseOptionList[k]) not in fileNames:
              fileNames[str(maxErrorIncreaseOptionList[k])] = []
            fileNames[str(maxErrorIncreaseOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[k], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 18:
          for k in range(len(maxLearningRateOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(maxLearningRateOptionList[k]) not in fileNames:
              fileNames[str(maxLearningRateOptionList[k])] = []
            fileNames[str(maxLearningRateOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[k], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 19:
          for k in range(len(minLearningRateOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(minLearningRateOptionList[k]) not in fileNames:
              fileNames[str(minLearningRateOptionList[k])] = []
            fileNames[str(minLearningRateOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[k], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 20:
          for k in range(len(annealingStartParmeterOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(annealingStartParmeterOptionList[k]) not in fileNames:
              fileNames[str(annealingStartParmeterOptionList[k])] = []
            fileNames[str(annealingStartParmeterOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[k], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 21:
          for k in range(len(annealingEndParmeterOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(annealingEndParmeterOptionList[k]) not in fileNames:
              fileNames[str(annealingEndParmeterOptionList[k])] = []
            fileNames[str(annealingEndParmeterOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[k], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 22:
          for k in range(len(weightDecayInUseOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(weightDecayInUseOptionList[k]) not in fileNames:
              fileNames[str(weightDecayInUseOptionList[k])] = []
            fileNames[str(weightDecayInUseOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[k], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 23:
          for k in range(len(batchLearningInUseOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(batchLearningInUseOptionList[k]) not in fileNames:
              fileNames[str(batchLearningInUseOptionList[k])] = []
            fileNames[str(batchLearningInUseOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[k], numberOfBatchesSplitIntoOptionList[0]))
        elif index == 24:
          for k in range(len(numberOfBatchesSplitIntoOptionList)):
            # If the option is not in the fileNames dictionary, add it
            if str(numberOfBatchesSplitIntoOptionList[k]) not in fileNames:
              fileNames[str(numberOfBatchesSplitIntoOptionList[k])] = []
            fileNames[str(numberOfBatchesSplitIntoOptionList[k])].append(returnFileName(headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], activationFunctionsOptionList[0], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[1], numberOfBatchesSplitIntoOptionList[k]))
  else:
    # fileNames in the form of:
    # {
    #   str(*option1*): [*fileNames*]
    # }
    fileNames = {}
    for i in range(len(weightsBiasesNumberOptionList)):
      for j in range(len(randomSplitNumberOptionList)):
        # If the option is not in the fileNames dictionary, add it
        if "Optimal" not in fileNames:
          fileNames["Optimal"] = []
        fileNames["Optimal"].append(returnFileName(headingsOptionList[1], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[1], activationFunctionsOptionList[6], numOfOutputNodesOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[1], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[1], boldDriverInUseOptionList[1], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]))

  return fileNames

def storePredictedVsActualFloodIndices(optionList, fileNames):
  # Read each file and store the weights and biases in lists with their original names
  for option in optionList:
    # For each file name, read the csv file and store the lists in their original file names
    for fileName in fileNames[str(option)]:
      with open(fileName, 'r', newline='') as file:
        reader = csv.reader(file)
        weightsFromInputToHidden = []
        weightsFromHiddenToOutput = []
        biasesFromInputToHidden = []
        biasesFromHiddenToOutput = []
        # Read the weights and biases from the file turning them into floats
        for row in reader:
          if row[0] == "# Weights from input to hidden nodes:":
            for row in reader:
              if row[0] == "# Weights from hidden to output nodes:":
                break
              # Append float values
              weightsFromInputToHidden.append([float(i) for i in row])
          if row[0] == "# Weights from hidden to output nodes:":
            for row in reader:
              if row[0] == "# Biases from input to hidden nodes:":
                break
              weightsFromHiddenToOutput.append([float(i) for i in row])
          if row[0] == "# Biases from input to hidden nodes:":
            biasesFromInputToHidden = [float(i) for i in next(reader)]
          if row[0] == "# Biases from hidden to output nodes:":
            biasesFromHiddenToOutput = [float(i) for i in next(reader)]
      
      fileParts = fileName.split(",")

      headingInfo = fileParts[0].split("_")[1]
      print(f"Heading info: {headingInfo}")
      headings = []

      # Given the following code generated headingInfo, the headings are:
      # for input in ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","Index flood"]:
      # if input in headings:
      #   fileName += "1"
      # else:
      #   fileName += "0"
      setSeadings = ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","Index flood"]
      for i in range(len(headingInfo)):
        if headingInfo[i] == "1":
          headings.append(setSeadings[i])
      print(f"Headings: {headings}")

      # Get the random split number
      randomSplitNumber = int(fileParts[10])
      print(f"Random split number: {randomSplitNumber}")

      # Set the directory of the split data
      if randomSplitNumber == -1:
        splitDataDirectory = "splitData/notRandom/"
      else:
        splitDataDirectory = f"splitData/random/dataSet{randomSplitNumber}/"

      # Get the activationFunctions list from the fileName
      activationFunctions = []
      activationFunctions.append(fileParts[3][1:])
      activationFunctions.append(fileParts[4][:-1])
      print(f"Activation functions: {activationFunctions}")

      # Function to load the data
      _, _, testingData = returnData(splitDataDirectory, headings)

      # Set the variables that are needed for the forward pass
      numOfInputNodes = len(headings) - 1
      numOfHiddenNodes = len(weightsFromInputToHidden[0])
      numOfOutputNodes = len(weightsFromHiddenToOutput[0])
      numOfHiddenLayers = 1

      actualVsPredictedFloodIndices = []

      # For each test data element work out the predicted flood index with the given weights and biases
      for row in testingData:
        input = [float(i) for i in row[:-1]]
        output = float(row[-1])
        # Forward Pass
        # Calculate the output of the hidden nodes and the output nodes
        hiddenOutput = [0 for i in range(numOfHiddenNodes)]
        for i in range(numOfHiddenNodes):
          hiddenOutput[i] = sum([input[j] * weightsFromInputToHidden[j][i] for j in range(numOfInputNodes)]) + biasesFromInputToHidden[i]
          if activationFunctions[0] == "sigmoid":
            hiddenOutput[i] = sigmoid(hiddenOutput[i])
          elif activationFunctions[0] == "tanh":
            hiddenOutput[i] = tanh(hiddenOutput[i])
          elif activationFunctions[0] == "relu":
            hiddenOutput[i] = max(0, hiddenOutput[i])
        
        finalOutputs = [0 for i in range(numOfOutputNodes)]
        for i in range(numOfOutputNodes):
          finalOutputs[i] = sum([hiddenOutput[j] * weightsFromHiddenToOutput[j][i] for j in range(numOfHiddenNodes)]) + biasesFromHiddenToOutput[i]
          if activationFunctions[numOfHiddenLayers] == "sigmoid":
            finalOutputs[i] = sigmoid(finalOutputs[i])
          elif activationFunctions[numOfHiddenLayers] == "tanh":
            finalOutputs[i] = tanh(finalOutputs[i])
          elif activationFunctions[numOfHiddenLayers] == "relu":
            finalOutputs[i] = max(0, finalOutputs[i])
        
        actualVsPredictedFloodIndices.append(f"{output}\t{finalOutputs[0]}")
      
      # Store the actual vs predicted flood indices in a csv file
      with open(f"aiData/actualVsPredictedFloodIndices/{fileName[29:]}.txt", 'w', newline='') as file:
        for line in actualVsPredictedFloodIndices:
          file.write(f"{line}\n")


def addToResults(results, title, optionList, fileNames):
  for option in optionList:
    # If the title is not in the dictionary, add it
    if title not in results:
      results[title] = {}
    # If the option is not in the dictionary, add it
    if str(option) not in results[title]:
      results[title][str(option)] = [[], [], [], []]

    # For each file name, read the csv file and store the lists in their original file names
    for fileName in fileNames[str(option)]:
      try:
        # Given that the file was stored using the code
        # with open(fileName, 'w', newline='') as file:
        #   writer = csv.writer(file)
        #   writer.writerow(["# Weights from input to hidden nodes:"])
        #   for row in weightsFromInputToHidden:
        #     writer.writerow(row)
        #   writer.writerow(["# Weights from hidden to output nodes:"])
        #   for row in weightsFromHiddenToOutput:
        #     writer.writerow(row)
        #   writer.writerow(["# Biases from input to hidden nodes:"])
        #   writer.writerow(biasesFromInputToHidden)
        #   writer.writerow(["# Biases from hidden to output nodes:"])
        #   writer.writerow(biasesFromHiddenToOutput)
        #   writer.writerow(["# Number of epochs trained for:"])
        #   writer.writerow([epochs])
        #   writer.writerow(["# Mean squared error of finished training:"])
        #   writer.writerow([testDataMeanError])
        #   writer.writerow(["# Number of backward pass calculations:"])
        #   writer.writerow([numOfBackwardPassCalculations])
        # Read the csv file and store the lists in their original file names
        with open(fileName, 'r', newline='') as file:
          reader = csv.reader(file)
          epochs = 0
          testDataMeanError = 0
          numOfBackwardPassCalculations = 0
          for row in reader:
            if row[0] == "# Number of epochs trained for:":
              epochs = int(next(reader)[0])
            if row[0] == "# Mean squared error of finished training:":
              testDataMeanError = float(next(reader)[0])
            if row[0] == "# Number of backward pass calculations:":
              numOfBackwardPassCalculations = int(next(reader)[0])

        # Store the information into the results dictionary
        # Add the list of values to the dictionary
        if len(results[title][str(option)][0]) < 3:
          results[title][str(option)][0].append([testDataMeanError, epochs, numOfBackwardPassCalculations])
        elif len(results[title][str(option)][1]) < 3:
          results[title][str(option)][1].append([testDataMeanError, epochs, numOfBackwardPassCalculations])
        elif len(results[title][str(option)][2]) < 3:
          results[title][str(option)][2].append([testDataMeanError, epochs, numOfBackwardPassCalculations])
        elif len(results[title][str(option)][3]) < 3:
          results[title][str(option)][3].append([testDataMeanError, epochs, numOfBackwardPassCalculations])
      except FileNotFoundError:
        print("File not found")
    
    while len(results[title][str(option)][0]) < 3:
      results[title][str(option)][0].append([0, 0, 0])
    while len(results[title][str(option)][1]) < 3:
      results[title][str(option)][1].append([0, 0, 0])
    while len(results[title][str(option)][2]) < 3:
      results[title][str(option)][2].append([0, 0, 0])
    while len(results[title][str(option)][3]) < 3:
      results[title][str(option)][3].append([0, 0, 0])

# Declare all test info from the aiRunner.py file
headingsOptionList = [
  ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","Index flood"],
  ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","SAAR","Index flood"],
  ["AREA","BFIHOST","LDP","PROPWET","SAAR","Index flood"],
  ["AREA","BFIHOST","LDP","PROPWET","Index flood"]
]
numOfHiddenLayersOptionList = [1]
numOfHiddenNodesInEachLayerOptionList = [[12], [8], [10], [14], [16]]
numOfOutputNodesOptionList = [1]
activationFunctionsOptionList = [["sigmoid", "sigmoid"], ["tanh", "tanh"], ["relu", "relu"], ["sigmoid", "tanh"], ["tanh", "relu"], ["relu", "sigmoid"], ["tanh", "sigmoid"]]
learningRateOptionList = [0.1]
maxEpochsOptionList = [10000]
frequencyOfValidationChecksOptionList = [500]
weightsBiasesNumberOptionList = [0, 1, 2]
randomSplitNumberOptionList = [0, 1, 2, 3]
momentumMultiplierInUseOptionList = [False, True]
momentumMultiplierOptionList = [0.9]
eitherBoldDriverOrAnnealingInUseOptionList = [False, True]
boldDriverInUseOptionList = [False,  True]
boldDriverIncreaseFactorOptionList = [1.05]
boldDriverDecreaseFactorOptionList = [0.7]
frequencyOfBoldDriverChecksOptionList = [10]
maxErrorIncreaseOptionList = [0.04]
maxLearningRateOptionList = [0.5]
minLearningRateOptionList = [0.01]
annealingStartParmeterOptionList = [0.1]
annealingEndParmeterOptionList = [0.01]
weightDecayInUseOptionList = [False, True]
batchLearningInUseOptionList = [False, True]
numberOfBatchesSplitIntoOptionList = [20, 1]

# Loop to read each file coming up with a title for the test, the MSE on the test data, the number of epochs taken, and the number of backwards pass calculations taken
# Store information into a dictionary with the title as the key and the rest as a list of values
# Format of the dictionary: 
# {
#   title: {
#     value: [
#       [[MSE, epochs, backwardsPasses], [MSE, epochs, backwardsPasses], [MSE, epochs, backwardsPasses]],
#       [[MSE, epochs, backwardsPasses], [MSE, epochs, backwardsPasses], [MSE, epochs, backwardsPasses]],
#       [[MSE, epochs, backwardsPasses], [MSE, epochs, backwardsPasses], [MSE, epochs, backwardsPasses]],
#       [[MSE, epochs, backwardsPasses], [MSE, epochs, backwardsPasses], [MSE, epochs, backwardsPasses]]
#     ]
#   }
# }

results = {}

if justOptimal == False:
  index = 0
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Headings Used", headingsOptionList, fileNames)
  storePredictedVsActualFloodIndices(headingsOptionList, fileNames)

  index = 1
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Number of Hidden Layers", numOfHiddenLayersOptionList, fileNames)

  index = 2
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Number of Hidden Nodes in Each Layer", numOfHiddenNodesInEachLayerOptionList, fileNames)

  index = 3
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Activation Functions", activationFunctionsOptionList, fileNames)

  index = 4
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Number of Output Nodes", numOfOutputNodesOptionList, fileNames)

  index = 5
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Learning Rate", learningRateOptionList, fileNames)

  index = 6
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Max Epochs", maxEpochsOptionList, fileNames)

  index = 7
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Frequency of Validation Checks", frequencyOfValidationChecksOptionList, fileNames)

  # index = 8
  # fileNames = returnFileNames(
  #   index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  # )
  # addToResults(results, "Weights and Biases Number", weightsBiasesNumberOptionList, fileNames)

  # index = 9
  # fileNames = returnFileNames(
  #   index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  # )
  # addToResults(results, "Random Split Number", randomSplitNumberOptionList, fileNames)

  index = 10
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Momentum Multiplier In Use", momentumMultiplierInUseOptionList, fileNames)

  index = 11
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Momentum Multiplier", momentumMultiplierOptionList, fileNames)

  index = 12
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Either Bold Driver or Annealing In Use", eitherBoldDriverOrAnnealingInUseOptionList, fileNames)

  index = 13
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Bold Driver In Use", boldDriverInUseOptionList, fileNames)

  index = 14
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Bold Driver Increase Factor", boldDriverIncreaseFactorOptionList, fileNames)

  index = 15
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Bold Driver Decrease Factor", boldDriverDecreaseFactorOptionList, fileNames)

  index = 16
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Frequency of Bold Driver Checks", frequencyOfBoldDriverChecksOptionList, fileNames)

  index = 17
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Max Error Increase", maxErrorIncreaseOptionList, fileNames)

  index = 18
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Max Learning Rate", maxLearningRateOptionList, fileNames)

  index = 19
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Min Learning Rate", minLearningRateOptionList, fileNames)

  index = 20
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Annealing Start Parameter", annealingStartParmeterOptionList, fileNames)

  index = 21
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Annealing End Parameter", annealingEndParmeterOptionList, fileNames)

  index = 22
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Weight Decay In Use", weightDecayInUseOptionList, fileNames)

  index = 23
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Batch Learning In Use", batchLearningInUseOptionList, fileNames)

  index = 24
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Number of Batches Split Into", numberOfBatchesSplitIntoOptionList, fileNames)
else:
  index = 0
  fileNames = returnFileNames(
    index, headingsOptionList, numOfHiddenLayersOptionList, numOfHiddenNodesInEachLayerOptionList, numOfOutputNodesOptionList, activationFunctionsOptionList, learningRateOptionList, maxEpochsOptionList, frequencyOfValidationChecksOptionList, weightsBiasesNumberOptionList, randomSplitNumberOptionList, momentumMultiplierInUseOptionList, momentumMultiplierOptionList, eitherBoldDriverOrAnnealingInUseOptionList, boldDriverInUseOptionList, boldDriverIncreaseFactorOptionList, boldDriverDecreaseFactorOptionList, frequencyOfBoldDriverChecksOptionList, maxErrorIncreaseOptionList, maxLearningRateOptionList, minLearningRateOptionList, annealingStartParmeterOptionList, annealingEndParmeterOptionList, weightDecayInUseOptionList, batchLearningInUseOptionList, numberOfBatchesSplitIntoOptionList
  )
  addToResults(results, "Presets", ["Optimal"], fileNames)
  storePredictedVsActualFloodIndices(["Optimal"], fileNames)

# results = {
#   "Momentum in Use": {
#     "True": [
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]]
#     ],
#     "False": [
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]]
#     ]
#   },
#   "One to Five": {
#     "1": [
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]]
#     ],
#     "2": [
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]]
#     ],
#     "3": [
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]]
#     ],
#     "4": [
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]]
#     ],
#     "5": [
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]],
#       [[0.005, 2500, 1050000], [0.005, 2500, 1050000], [0.005, 2500, 1050000]]
#     ]
#   }
# }

# Store the results in a .txt file in a format that can be copy and pasted into Excel
excelDoc = [
  "",
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", # 16
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
]

for title in results:
  excelDoc[0] += f"\t\t\t{title}\t\t\t"
  for valueIndex, value in enumerate(results[title]):
    excelDoc[1 + 16 * valueIndex] += f"\t\t\t{value}\t\t\t"
    excelDoc[2 + 16 * valueIndex] += f"\t\t\tWeights and Biases Start File Number\t\t\t"
    excelDoc[3 + 16 * valueIndex] += f"\t\t0\t1\t2\t\t"
    for resultIndex, result in enumerate(results[title][value]):
      # print(result, title)
      if resultIndex == 2:
        excelDoc[4 + 16 * valueIndex + 3 * resultIndex] += f"File\t\t{result[0][0]}\t{result[1][0]}\t{result[2][0]}\t\t"
      else:
        excelDoc[4 + 16 * valueIndex + 3 * resultIndex] += f"\t\t{result[0][0]}\t{result[1][0]}\t{result[2][0]}\t\t"
      if resultIndex == 1:
        excelDoc[5 + 16 * valueIndex + 3 * resultIndex] += f"Random\t{resultIndex}\t{result[0][1]}\t{result[1][1]}\t{result[2][1]}\t\t"
      elif resultIndex == 2:
        excelDoc[5 + 16 * valueIndex + 3 * resultIndex] += f"Number\t{resultIndex}\t{result[0][1]}\t{result[1][1]}\t{result[2][1]}\t\t"
      else:
        excelDoc[5 + 16 * valueIndex + 3 * resultIndex] += f"\t{resultIndex}\t{result[0][1]}\t{result[1][1]}\t{result[2][1]}\t\t"
      if resultIndex == 1:
        excelDoc[6 + 16 * valueIndex + 3 * resultIndex] += f"Split\t\t{result[0][2]}\t{result[1][2]}\t{result[2][2]}\t\t"
      else:
        excelDoc[6 + 16 * valueIndex + 3 * resultIndex] += f"\t\t{result[0][2]}\t{result[1][2]}\t{result[2][2]}\t\t"
    excelDoc[16 + 16 * valueIndex] += "\t\t\t\t\t\t"
  leftOver = 7 - len(results[title])
  for leftOverIndex in range(leftOver):
    for i in range(16):
      excelDoc[1 + 16 * (len(results[title]) + leftOverIndex) + i] += "\t\t\t\t\t\t"

    

# excelDoc = [
#   "\t\t\t*Variable*\t\t\t",
#   "\t\t\t*Value*\t\t\t",
#   "\t\t\tWeights and Biases Start File Number\t\t\t",
#   "\t\t0\t1\t2\t\t",
#   "\t\t*MSE0*\t*MSE1*\t*MSE2*\t\t",
#   "\t0\t*Epochs0*\t*Epochs1*\t*Epochs2*\t\t",
#   "\t\t*Back0*\t*Back1*\t*Back2*\t\t",
#   "\t\t*MSE0*\t*MSE1*\t*MSE2*\t\t",
#   "Random\t1\t*Epochs0*\t*Epochs1*\t*Epochs2*\t\t",
#   "Split\t\t*Back0*\t*Back1*\t*Back2*\t\t",
#   "File\t\t*MSE0*\t*MSE1*\t*MSE2*\t\t",
#   "Number\t2\t*Epochs0*\t*Epochs1*\t*Epochs2*\t\t",
#   "\t\t*Back0*\t*Back1*\t*Back2*\t\t",
#   "\t\t*MSE0*\t*MSE1*\t*MSE2*\t\t",
#   "\t3\t*Epochs0*\t*Epochs1*\t*Epochs2*\t\t",
#   "\t\t*Back0*\t*Back1*\t*Back2*\t\t",
#   "\t\t\t\t\t\t",
#   "\t\t\t*Value*\t\t\t",
#   "\t\t\tWeights and Biases Start File Number\t\t\t",
#   "\t\t0\t1\t2\t\t",
#   "\t\t*MSE0*\t*MSE1*\t*MSE2*\t\t",
#   "\t0\t*Epochs0*\t*Epochs1*\t*Epochs2*\t\t",
#   "\t\t*Back0*\t*Back1*\t*Back2*\t\t",
#   "\t\t*MSE0*\t*MSE1*\t*MSE2*\t\t",
#   "Random\t1\t*Epochs0*\t*Epochs1*\t*Epochs2*\t\t",
#   "Split\t\t*Back0*\t*Back1*\t*Back2*\t\t",
#   "File\t\t*MSE0*\t*MSE1*\t*MSE2*\t\t",
#   "Number\t2\t*Epochs0*\t*Epochs1*\t*Epochs2*\t\t",
#   "\t\t*Back0*\t*Back1*\t*Back2*\t\t",
#   "\t\t*MSE0*\t*MSE1*\t*MSE2*\t\t",
#   "\t3\t*Epochs0*\t*Epochs1*\t*Epochs2*\t\t",
#   "\t\t*Back0*\t*Back1*\t*Back2*\t\t",
#   "\t\t\t\t"
# ]

# for index, line in enumerate(excelDoc):
#   excelDoc[index] += line

# Open the file that stores the results
with open('aiData/excelFormattedResults.txt', 'w') as file:
  for line in excelDoc:
    file.write(line + '\n')