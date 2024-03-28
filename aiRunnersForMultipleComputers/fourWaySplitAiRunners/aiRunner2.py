# Import the local file ai.py
import ai

# Create lists of possible values for each of the parameters: headings, numOfHiddenLayers, numOfHiddenNodesInEachLayer, numOfOutputNodes, activationFunctions, learningRate, maxEpochs, frequencyOfValidationChecks, weightsBiasesNumber, randomSplitNumber, momentumMultiplierInUse, momentumMultiplier, eitherBoldDriverOrAnnealingInUse, boldDriverInUse, boldDriverIncreaseFactor, boldDriverDecreaseFactor, frequencyOfBoldDriverChecks, maxErrorIncrease, maxLearningRate, minLearningRate, annealingStartParmeter, annealingEndParmeter, weightDecayInUse, batchLearningInUse, numberOfBatchesSplitInto
# The default values are the first values in each list
weightsBiasesNumberOptionList = [0, 1, 2]
randomSplitNumberOptionList = [2]

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

# Run the main function in ai.py with the default values for each parameter apart from one parameter at a time go through the list
for i in range(len(weightsBiasesNumberOptionList)):
  for j in range(len(randomSplitNumberOptionList)):
    # Run default values for all parameters
    result1, result2 = ai.main(
      headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
    )
    thisNum = 1
    if result1 != None:
      print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    elif result1 == None:
      print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    # For each parameter, run the main function with the non-default values for that parameter
    for k in range(1, len(headingsOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[k], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 2
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(numOfHiddenLayersOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[k], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 3
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(numOfHiddenNodesInEachLayerOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[k], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 4
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(activationFunctionsOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[k], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 5
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(learningRateOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[k], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 6
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(maxEpochsOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[k], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 7
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(frequencyOfValidationChecksOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[k], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 8
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(momentumMultiplierInUseOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[k], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 9
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(momentumMultiplierOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[k], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 10
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")

    # eitherBoldDriverOrAnnealingInUseOptionList affects boldDriverInUseOptionList
    for k in range(1, len(eitherBoldDriverOrAnnealingInUseOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[k], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 11
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    
      if eitherBoldDriverOrAnnealingInUseOptionList[k] == True:
        for l in range(1, len(boldDriverInUseOptionList)):
          result1, result2 = ai.main(
            headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[k], boldDriverInUseOptionList[l], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
          )
          thisNum = 12
          if result1 != None:
            print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
          if result1 == None:
            print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")

    for k in range(1, len(boldDriverIncreaseFactorOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[k], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 13
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(boldDriverDecreaseFactorOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[k], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 14
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(frequencyOfBoldDriverChecksOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[k], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 15
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(maxErrorIncreaseOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[k], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 16
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(maxLearningRateOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[k], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 17
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(minLearningRateOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[k], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 18
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(annealingStartParmeterOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[k], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 19
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(annealingEndParmeterOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[k], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 20
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(weightDecayInUseOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[k], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 21
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(batchLearningInUseOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[k], numberOfBatchesSplitIntoOptionList[0]
      )
      thisNum = 22
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    for k in range(1, len(numberOfBatchesSplitIntoOptionList)):
      result1, result2 = ai.main(
        headingsOptionList[0], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[0], numOfOutputNodesOptionList[0], activationFunctionsOptionList[0], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[0], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[0], boldDriverInUseOptionList[0], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[1], numberOfBatchesSplitIntoOptionList[k]
      )
      thisNum = 23
      if result1 != None:
        print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
      if result1 == None:
        print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    # Run default values for all parameters
    result1, result2 = ai.main(
      headingsOptionList[1], numOfHiddenLayersOptionList[0], numOfHiddenNodesInEachLayerOptionList[1], numOfOutputNodesOptionList[0], activationFunctionsOptionList[6], learningRateOptionList[0], maxEpochsOptionList[0], frequencyOfValidationChecksOptionList[0], weightsBiasesNumberOptionList[i], randomSplitNumberOptionList[j], momentumMultiplierInUseOptionList[1], momentumMultiplierOptionList[0], eitherBoldDriverOrAnnealingInUseOptionList[1], boldDriverInUseOptionList[1], boldDriverIncreaseFactorOptionList[0], boldDriverDecreaseFactorOptionList[0], frequencyOfBoldDriverChecksOptionList[0], maxErrorIncreaseOptionList[0], maxLearningRateOptionList[0], minLearningRateOptionList[0], annealingStartParmeterOptionList[0], annealingEndParmeterOptionList[0], weightDecayInUseOptionList[0], batchLearningInUseOptionList[0], numberOfBatchesSplitIntoOptionList[0]
    )
    thisNum = 24
    if result1 != None:
      print(f"Program run at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")
    if result1 == None:
      print(f"Already exists at {thisNum} with weightsBiasesNumberOptionList {weightsBiasesNumberOptionList[i]} and randomSplitNumberOptionList {randomSplitNumberOptionList[j]}.")