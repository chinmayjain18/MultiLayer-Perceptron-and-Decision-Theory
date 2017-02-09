# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:49:02 2015

@author: Chinmay Jain
@author: Tolga Cerrah

Project 2: SORT-R
Using Decision Tree classification for metal piece classification
Format: symmetry,eccentricity,output_class
"""
import numpy
import pickle
from pylab import matplotlib as plotter

class Node:
    '''
    Node : Represents a node in the binary tree
    '''
    def __init__(self,info,attribute):
        self.info = info;
        self.attributeName = attribute;
        self.left = '';
        self.right = '';
        self.ACount = '';
        self.BCount = '';
        self.CCount = '';
        self.DCount = '';

class BinaryTree:
    '''
    Binary Tree :  Represents a binary tree of Nodes
    '''
    def __init__ (self,root,attribute):
        self.root = Node(root,attribute);
        
class Count:
    def __init__ (self, nodeCount, leafCount):
        self.nodeCount = nodeCount;
        self.leafCount = leafCount;

def getMode(outputclass, instances):
    count = [0]*(len(outputclass));
    for instance in instances:
        for j in range(len(outputclass)):        
            if instance[-1] ==  outputclass[j]:
                count[j] += 1;
                break;
    max = 0;
    mode = 0;
    for i in range(len(outputclass)):
        if count[i] > max:
            max = count[i];
            mode = i;
    return mode;
    
def informationGain(x,attribute,instances):
    n = float(len(instances));
    count = [0]*8;
    countL = 0;
    countR = 0;
    outCount = [0]*4;
    for i in range(len(instances)):
        outCount[int(instances[i][2])-1] += 1;
    infoGainRoot = 0.0;
    for i in range(4):
        if outCount[i] != 0:
            infoGainRoot += -(outCount[i]/n)*numpy.log2(outCount[i]/n);
    
    for i in range(len(instances)):
        if float(instances[i][attribute]) >= x:
            count[int(instances[i][2])-1] += 1;
            countR += 1;
        else:
            count[int(instances[i][2])-1+4] += 1;
            countL += 1;
    infoGainR = 0.0;
    infoGainL = 0.0;
    for i in range(4):
        if countR != 0 and count[i] != 0:
            infoGainR += -(count[i]/float(countR))* numpy.log2(count[i]/float(countR));
        if countL != 0 and count[i+4] != 0:
            infoGainL += -(count[i+4]/float(countL))* numpy.log2(count[i+4]/float(countL));
    infoGain = 0.0;
    if infoGainR is not None:
        infoGain += (countR/n)*infoGainR;
    if infoGainL is not None:
        infoGain += (countL/n)*infoGainL;
        
    infoGain = infoGainRoot - infoGain;
    return infoGain;
    
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

def plotBoundaries(tree):
    xaxis = [];
    yaxis = [];
    outputList = [];
    z = numpy.arange(0,1.05,0.01);
    for x in z:
        for y in z:
            xaxis.append(x);
            yaxis.append(y);
            outputClass = getOutputclass([x,y],tree);
            outputList.append(outputClass);
    plotter.pyplot.xlim(0,1);
    plotter.pyplot.ylim(0,1);
    plotter.pyplot.scatter(xaxis, yaxis,c=outputList);
    plotter.pyplot.show();
    
def DLTTrain(instances, attributes, default):
    if not instances:
        #print('Case1');
        return default;
    else:
        outputclass = [];
        # Counts of all the classes in this node
        countA = 0;
        countB = 0;
        countC = 0;
        countD = 0;
        for i in range(len(instances)):
            tempClass = instances[i][2];
            if (tempClass[0] == '1'):
                countA += 1;
            elif (tempClass[0] == '2'):
                countB += 1;
            elif (tempClass[0] == '3'):
                countC += 1;
            else:
                countD += 1;
            outputclass.append(tempClass);
        outputclass = set(outputclass);
        if len(outputclass) == 1:
            #print('Case2');
            tree = BinaryTree(list(outputclass)[0][0], len(instances))
            tree.root.ACount = countA;
            tree.root.BCount = countB;
            tree.root.CCount = countC;
            tree.root.DCount = countD;
            return tree
        elif not attributes:
            #print('Case3');
            return getMode(outputclass, instances);
        else:
            #print('Case4');
            maxValue = 0;
            splitAttribute = 0;
            splitValue = 0;
            for j in range(len(attributes)):
                instances.sort(key=lambda x:x[attributes[j]]);
                for i in range(len(instances)-1):
                    mid1 = (float(instances[i][j])+float(instances[i+1][j]))/2;
                    infoGain = informationGain(mid1,j,instances);
                    if infoGain > maxValue:
                        maxValue = infoGain;
                        splitAttribute = j;
                        splitValue = mid1;
            tree = BinaryTree(splitValue,splitAttribute);
            rightInstances = [];
            leftCount = 0;
            rightCount = 0;
            leftInstances =[];
            for i in range(len(instances)):
                if float(instances[i][splitAttribute]) >= splitValue:
                    rightInstances.append(instances[i]);
                    rightCount += 1;
                else:
                    leftInstances.append(instances[i]);
                    leftCount += 1;
            tree.root.ACount = countA;
            tree.root.BCount = countB;
            tree.root.CCount = countC;
            tree.root.DCount = countD;
            tree.root.left = DLTTrain(leftInstances, attributes, default);
            tree.root.right = DLTTrain(rightInstances, attributes, default);
    return tree;

def printTree(tree, level):
    s = ''
    for i in range(level):
        s += '\t'
    if not tree:
        print(s,'')
    else:
        print(s, 'Node: ', tree.root.info)
        print(s, 'Attribute', tree.root.attributeName + 1)
        print(s, 'A count', tree.root.ACount);
        print(s, 'B count', tree.root.BCount);
        print(s, 'C count', tree.root.CCount);
        print(s, 'D count', tree.root.DCount);
        if (not tree.root.left and not tree.root.right):
            print(s, 'This is a leaf!\n');
        else:
            printTree(tree.root.left, level+1)
            printTree(tree.root.right, level+1)
            
def outputSum(treeNode):
    return treeNode.ACount + treeNode.BCount + treeNode.CCount + treeNode.DCount;

def dof(node):
    outClasses = [node.ACount, node.BCount, node.CCount, node.DCount];
    dofV = -1;
    for n in outClasses:
        if n > 0:
            dofV += 1;
    return dofV;

def chiSquare(tree):
    parent = tree.root
    left = parent.left.root
    right = parent.right.root
    
    delta = 0.0
    #Expected value for the left node
    outputSumP = outputSum(parent);
    pA = parent.ACount/outputSumP;
    pB = parent.BCount/outputSumP;
    pC = parent.CCount/outputSumP;
    pD = parent.DCount/outputSumP;
    
    outputSumL = outputSum(left);
    outputSumR = outputSum(right);
    
    if pA != 0.0:    
        delta += numpy.square((left.ACount - outputSumL * pA))/(outputSumL * pA)
        delta += numpy.square((right.ACount - outputSumR * pA))/(outputSumR * pA)
    
    if pB != 0.0:
        delta += numpy.square((left.BCount - outputSumL * pB))/(outputSumL * pB)
        delta += numpy.square((right.BCount - outputSumR * pB))/(outputSumR * pB)
    pacC = 0;
    if pC != 0.0:    
        delta += numpy.square((left.CCount - outputSumL * pC))/(outputSumL * pC)
        delta += numpy.square((right.CCount - outputSumR * pC))/(outputSumR * pC)
        pacC += numpy.square((left.CCount - outputSumL * pC))/(outputSumL * pC);
        pacC += numpy.square((right.CCount - outputSumR * pC))/(outputSumR * pC)
    pacD = 0;
    if pD != 0.0:    
        delta += numpy.square((left.DCount - outputSumL * pD))/(outputSumL * pD)
        delta += numpy.square((right.DCount - outputSumR * pD))/(outputSumR * pD)
        pacD += numpy.square((left.DCount - outputSumL * pD))/(outputSumL * pD)
        pacD += numpy.square((right.DCount - outputSumR * pD))/(outputSumR * pD)
    
    dofV = dof(tree.root); 
    # 5% significance level
    if dofV == 1:
        threshold = 3.84;
    elif dofV == 2:
        threshold = 5.99;
    else:
        threshold = 7.81;
    if (delta > threshold):
        #print('Pruned')
        print('Delta',delta);
        print('PC',pacC);
        print('PD',pacD);
        return True
    else:
        #print('NotPruned')
        return False

def getMode2(tree):
    modeClass = 1
    modeClassVal = tree.root.ACount
    if (tree.root.BCount > modeClassVal):
        modeClass = 2
        modeClassVal = tree.root.BCount
    if (tree.root.CCount > modeClassVal):
        modeClass = 3
        modeClassVal = tree.root.CCount
    if (tree.root.DCount > modeClassVal):
        modeClass = 4
        modeClassVal = tree.root.DCount
    return modeClass

def traverseAndCheck(tree):
    if (not tree.root.left and not tree.root.right):
        return True
    else:
        leftCheck = traverseAndCheck(tree.root.left);
        rightCheck = traverseAndCheck(tree.root.right);
        if (leftCheck and rightCheck):
            pruned = chiSquare(tree) #pruned            
            if pruned == True:
                #print('tree.root.info',tree.root.info)
                tree.root.info = getMode2(tree)
                tree.root.attribute = tree.root.info
                tree.root.left = ''
                tree.root.right = ''           
                return True
        return False

def getTreeStats(tree):
    count = Count(0,0);
    def treeDepths(tree, depth):
        count.nodeCount += 1;
        if (not tree.root.left and not tree.root.right):
            count.leafCount += 1;
            return [depth]
        else:
            return treeDepths(tree.root.left, depth+1) + treeDepths(tree.root.right, depth+1);
    
    depths = treeDepths(tree, 1);
    print('Number of nodes: ',count.nodeCount);
    print('Number of leaves: ', count.leafCount);
    print('Max depth: ', max(depths));
    print('Min depth: ', min(depths));
    print('Average depth:', sum(depths)/len(depths));  

def main():
    file = open('train_data.csv','r');
    instances = file.readlines();
    data = [[0 for x in range(3)] for x in range(len(instances))];
    for i in range(len(instances)):
        x = instances[i].split(',');
        data[i][0] = x[0];
        data[i][1] = x[1];
        data[i][2] = x[2];
    tree = DLTTrain(data, [0,1], 4);
    print('Stats of the tree before pruning:');
    getTreeStats(tree);
    print('Decision regions for decision tree');
    plotBoundaries(tree);
    printTree(tree, 0);
    pickle.dump(tree,open('DecsionTree.pkl','wb'));
    traverseAndCheck(tree);
    print('Stats of the tree after pruning:');
    getTreeStats(tree);
    print('Decision regions for decision tree after pruning');        
    plotBoundaries(tree);
    #printTree(tree,0);
    pickle.dump(tree,open('DecsionTreePruned.pkl','wb'));
 
main();
