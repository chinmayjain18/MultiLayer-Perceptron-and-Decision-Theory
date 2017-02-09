# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:05:12 2015

@author: Chinmay Jain
@author: Tolga Cerrah

Test execution file for the trained MLP model
"""
import argparse
import math
from pylab import matplotlib as plotter

def roundFloat(a):
    return (float)("{0:.2f}".format(a));

def getSigmoidValue(z):
    return (1.0 /(1.0 + math.exp(-z)));
    
def getOutputClass(x):
    maxValue = max(x);
    for i in range(len(x)):
        if x[i] == maxValue:
            return i;
            

def calculateProfit(cf):
    profit = 0;
    pm = [[20,-7,-7,-7],[-7,15,-7,-7],[-7,-7,5,-7],[-3,-3,-3,-3]];
    for i in range(4):
        for j in range(4):
            profit += cf[i][j]*pm[i][j];
    print('Profit=',profit);

def main():
    parser = argparse.ArgumentParser(description='Execute MLP for testing file');
    parser.add_argument('weights_File', help='weight to be used');
    args = parser.parse_args();
    
    cf = [[0 for x in range(4)] for x in range(4)];    
    
    weightsFile = open(args.weights_File,'r');
    weights = weightsFile.read();
    weights = weights.split(',');
    weightsLayer1 = [];
    weightsLayer2 = [];
    for i in range(11):
        weightsLayer1.append(float(weights[i]));
    for i in range(21):
        weightsLayer2.append(float(weights[i+11]));
    
    testFile = open('test_data.csv','r');
    data = testFile.readlines();
    acc = 0;
    xaxis = [];
    yaxis = [];
    outputClassList = [];
    for i in range(len(data)):
        x = (data[i]).split(',');
        xaxis.append(x[0]);
        yaxis.append(x[1]);
        h = [weightsLayer1[10]]*5;
        for i in range(5):
            h[i] += weightsLayer1[i]*float(x[0]) + weightsLayer1[i+5]*float(x[1]);
            h[i] = getSigmoidValue(h[i]);
        o = [weightsLayer2[20]]*4;
        for i in range(4):
            o[i] = h[0]*weightsLayer2[i] +h[1]*weightsLayer2[i+4] +h[2]*weightsLayer2[i+8] +h[3]*weightsLayer2[i+12] +h[4]*weightsLayer2[i+16];
        outputClass = getOutputClass(o);
        #print(outputClass,x[2]);
        if (outputClass+1) == int(x[2]):
            cf[outputClass][outputClass] += 1;
            acc += 1;
        else:
            cf[outputClass][int(x[2])-1] += 1;
        outputClassList.append(outputClass+1);
    acc = float(acc/len(data)) * 100;
    print('Testing Accuracy=',acc,'%');
    print('Confusion Matrix=');
    for i in range(4):
        print(cf[i][0],cf[i][1],cf[i][2],cf[i][3]);
    calculateProfit(cf);
    plot.xlim(0,1);
    plot.ylim(0,1);
    print('Classification Regions');
    plot.scatter(xaxis, yaxis,c=outputClassList);
    
plot = plotter.pyplot;
main()