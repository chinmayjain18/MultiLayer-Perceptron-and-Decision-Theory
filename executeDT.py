# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:22:43 2015

@author: Chinmay Jain
@author: Tolga Cerrah
"""
import pickle
from pylab import matplotlib as plot

def getOutputclass(x, tree):
    temp = tree;
    #print(temp);
    outputclass = 0;
    while (outputclass == 0):
        if (not temp.root.left and not temp.root.right):
            outputclass = int(temp.root.info);
        else:
            attribute = temp.root.attributeName;
            splitValue = temp.root.info;
            #print('Testing attribute',attribute,'with value',splitValue);
            if float(x[attribute]) >= splitValue:
                temp = temp.root.right;
                #print('Going right');
            else:
                temp = temp.root.left;
                #print('Going left');
    return outputclass;

def calculateProfit(cf):
    profit = 0;
    pm = [[20,-7,-7,-7],[-7,15,-7,-7],[-7,-7,5,-7],[-3,-3,-3,-3]];
    for i in range(4):
        for j in range(4):
            profit += cf[i][j]*pm[i][j];
    print('Profit=',profit);

def testTree(tree,data):
    acc = 0;
    cf = [[0 for x in range(4)] for x in range(4)]; 
    outputClassList = [];
    xaxis = [];
    yaxis = [];
    for x in data:
        xaxis.append(x[0]);
        yaxis.append(x[1]);
        outputClass = getOutputclass(x,tree);
        outputClassList.append(outputClass);
        if outputClass == int(x[2]):
            cf[outputClass-1][outputClass-1] += 1;
            acc += 1;
        else:
            cf[outputClass-1][int(x[2])-1] += 1;
    acc = acc/len(data) * 100;
    print('Training Accuracy=',acc);
    print('Confusion Matrix=');
    for i in range(4):
        print(cf[i][0],cf[i][1],cf[i][2],cf[i][3]);
    calculateProfit(cf);
    # Plot decision boundaries
    plot.pyplot.xlim(0,1);
    plot.pyplot.ylim(0,1);
    print('Classification boundaries');
    plot.pyplot.scatter(xaxis, yaxis,c=outputClassList);
    plot.pyplot.show();

def main():
    # Read the test data
    file = open('test_data.csv','r');
    instances = file.readlines();
    data = [[0 for x in range(3)] for x in range(len(instances))];
    for i in range(len(instances)):
        x = instances[i].split(',');
        data[i][0] = x[0];
        data[i][1] = x[1];
        data[i][2] = x[2];
    # Testing beginns here    
    tree = pickle.load(open('DecsionTree.pkl','rb')); 
    print('Without Chi Square Pruning');
    testTree(tree,data);
    # After Chi Square Pruning
    print('_________________________________________________________________');
    print('With Chi Square Pruning');
    prunedTree = pickle.load(open('DecsionTreePruned.pkl','rb'));
    testTree(prunedTree,data);
    
main()