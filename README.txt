trainMLP.py
This file does not take any input arguments.
It trains the MLP network and write the training weights after 10,100, 1000 and 10000 epochs to their respective csv files.
It outputs the classification accuracy and confusion matrix after each of the epochs counts mentioned above.
Finally after the 10000th epoch it plots the learning curve before finishing the execution.

executeMLP.py
This file takes an input argument, the weights file to be used. 
command for execution : python executeMLP.py weights_after_10_epochs.csv.
This will read the weights from the csv file and test the data from the test_data.csv.
After execution the code will print the classification accuracy, confusion matrix, profit and the classification plot for the test data points.

trainDT.py
This file does not take any input arguments.
It defines a decision tree and stores it in the picke file- DecsionTree.pkl
Then the tree undergoes Chi Square pruning and the pruned tree is stored in - DecsionTreePruned.pkl
Finally the code will plot the decision regions and the tree details required.

executeDT.py
This file does not take any input arguments.
It reads two decision tree from DecsionTree.pkl and DecsionTreePruned.pkl.
It performs classification using the data from test_data.csv and the two tree read above.
Now it outputs the classification accuracy, confusion matrix, profit and the classification plots for the test data.