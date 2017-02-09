"""
Created on Mon Nov  9 15:22:35 2015

@author: Chinmay Jain
@author: Tolga Cerrah

Project 2: SORT-R
Using MLP classification for metal piece classification
Format: symmetry,eccentricity,output_class
"""
import random
import math
import numpy
from pylab import matplotlib as plotter

def roundFloat(a):
    return (float)("{0:.2f}".format(a));

def loadData(filename):
    file = open(filename,'r');
    data = file.readlines();
    return data;

def initializeWeights(weightsList):
    for i in range(len(weightsList)):
        weightsList[i] = (roundFloat(random.random()));
        sign = random.randrange(-1,2);
        if sign == 0:
            weightsList[i] = 0;
        else:
            weightsList[i] = weightsList[i] * sign;
    weightsList[-1] = 1;
    return weightsList;

def getSigmoidValue(z):
    return (1.0 /(1.0 + math.exp(-z)));

def writeFile(fileName, weights):
    file = open(fileName,'w');
    line = '';
    for weight in weights:
        line += str(weight) + ",";
    file.write(line);
    file.close();

def getOutputClass(x):
    maxValue = max(x);
    for i in range(len(x)):
        if x[i] == maxValue:
            return i;

def plotBoundaries(weights):
    weights1 = weights['Layer1'];
    weights2 = weights['Layer2'];
    z = numpy.arange(0,1.05,0.01);
    xaxis = [];
    yaxis = [];
    outputList = [];
    for x in z:
        for y in z:
            xaxis.append(x);
            yaxis.append(y);
            h = [weights1[10]]*5;
            for i in range(5):
                h[i] += weights1[i]*x + weights1[i+5]*y;
                h[i] = getSigmoidValue(h[i]);
            o = [weights2[20]]*4;
            for i in range(4):
                o[i] = h[0]*weights2[i] +h[1]*weights2[i+4] +h[2]*weights2[i+8] +h[3]*weights2[i+12] +h[4]*weights2[i+16];
            outputClass = getOutputClass(o);
            outputList.append(outputClass);
    plotter.xlim(0,1);
    plotter.ylim(0,1);
    plotter.scatter(xaxis, yaxis,c=outputList);
    plotter.show();
            
def train(epochCount, data):
    alpha = 0.1;
    weightsDict = {'Layer1':([0]*11),'Layer2':([0]*21)};
    weightsDict['Layer1'] = initializeWeights(weightsDict['Layer1']);
    weightsDict['Layer2'] = initializeWeights(weightsDict['Layer2']);
    fileName = 'weights_after_0_epochs.csv';
    weights = weightsDict['Layer1'];
    weights = weights + (weightsDict['Layer2']);
    writeFile(fileName, weights);
    print('Initial weights written to the file',fileName);
    print('_________________________________________________________');
    # Error SSD list for plotting a graph later
    xaxis = [];
    errorSSDPlotList = [];
    for count in range(epochCount):
        acc  = 0;
        cf = [[0 for x in range(4)] for x in range(4)];
        errorSSDList = [];
        for instance in data:
            x = instance.split(',');
            # Hidden Layer Processing
            weights = weightsDict['Layer1'];            
            h = [(int)(weights[-1])]*5;            
            hs = [0]*5;
            for i in range(5):
                h[i] += (weights[i])*(float)(x[0]) + (weights[i+5])*(float)(x[1]);
            for i in range(5):
                hs[i] = getSigmoidValue(roundFloat(h[i]));
            # Output Layer Processing
            weights = weightsDict['Layer2'];
            o = [weights[-1]]*4;
            os = [0]*4;
            output = [0]*4;
            for i in range(4):
                o[i] += weights[i]*hs[0] + weights[i+4]*hs[1] + weights[i+8]*hs[2] + weights[i+12]*hs[3] + weights[i+16]*hs[4];
                os[i] = getSigmoidValue(roundFloat(o[i]));
            outputClass = getOutputClass(os);
            output[outputClass] = 1;
            # Calculating Error
            error = [0]*4;
            errorSSD = [0]*4;
            for i in range(4):
                if int(x[2]) == (i+1):
                    error[i] = os[i] - 1;
                else:
                    error[i] = os[i];
                errorSSD[i] = error[i]*os[i]*(1-os[i]);
            # Setting up Error SSD plot - Learning Curve
            errorSSDList.append(sum(errorSSD));
            xaxis.append(count);
            # Increase hit count if output matches
            if (outputClass+1) == int(x[2]):
                cf[outputClass][outputClass] += 1;                
                acc += 1;
            else:
                cf[outputClass][int(x[2])-1] += 1;
            # Update weights using backpropagation using alpha
            # Update Hidden Layer Weights
            weights = weightsDict['Layer1'];
            weights2 = weightsDict['Layer2'];
            for j in range(2):
                for i in range(5):
                    delta = 0;
                    for k in range(4):
                        delta += weights2[k+(i*4)] * errorSSD[k];
                    delta = delta * hs[i] * (1-hs[i]);
                    weights[j*5+i] -= alpha * (float)(x[j]) * delta;
            # Update Output Layer Weights
            weights = weightsDict['Layer2'];
            for j  in range(5):
                for i in range(4):
                    weights[j*4+i] -= alpha * errorSSD[i] * hs[j];
        if (count+1) in [10,100,1000,10000]:
            fileName = 'weights_after_'+str(count+1)+'_epochs.csv';
            weights = weightsDict['Layer1'];
            weights = weights + (weightsDict['Layer2']);
            writeFile(fileName, weights);
            print('Writing weights to file',fileName);
            acc = ((float)(acc))/(len(data)) * 100;
            print('Training accuracy after',(count+1),'epochs is',roundFloat(acc),'%');
            print('Confusion Matrix=');
            for i in range(4):
                print(cf[i][0],cf[i][1],cf[i][2],cf[i][3]);
            plotBoundaries(weightsDict);
            print('_________________________________________________________');
        errorSSDPlotList.append(sum(errorSSDList));
    print('Learning curve');
    plotter.plot(errorSSDPlotList);
    plotter.show();
        
def main():
    trnData = loadData('train_data.csv');
    epochs = [10000];
    for epoch in epochs:
        train(epoch, trnData);    
    
plotter = plotter.pyplot;
main()