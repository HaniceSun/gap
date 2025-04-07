import sys
import os
import colorcet
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import pylab as plt

cout = ['#ffd845', '#710027', '#778500', '#018ead', '#c44cfd']
cin = sns.color_palette('tab20') + sns.color_palette('Set2')

def DonutPlot(inF, DIR='Plots'):
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    ouFile = open(DIR + '/' + DIR + '_Prediction.log', 'w')
    ouFile.write('\t'.join(['Sample', 'SuperPopulation', 'Population', 'Consistency']) + '\n')
    inFile = open(inF)
    head = inFile.readline()
    for line in inFile:
        line = line.strip()
        fields = line.split('\t')
        sample = fields[0]
        super_population_class = fields[1]
        population_class = fields[2]
        super_population_possibility = [x.split(':') for x in fields[4].split('|')]
        population_possibility = [x.split(':') for x in fields[5].split('|')]
        consistency = fields[3]

        ouFile.write('\t'.join([sample, super_population_class, population_class, consistency]) + '\n')
        if sample.find('HIPP') != -1:
            sample = sample.split('_')[-1].split('DNA')[0]

        fig, ax = plt.subplots()
        ax.axis('equal')
        width = 0.3

        probs = [float(x[1]) for x in super_population_possibility]
        labels = [x[0] for x in super_population_possibility]
        label_max = labels[probs.index(max(probs))]
        if consistency == 'Consistent':
            labels2 = [x if x == label_max else '' for x in labels]
        else:
            labels2 = [x + '?' if x == label_max else '' for x in labels]
        pie, _ = ax.pie(probs, radius=1, labels=labels2, colors=cout, labeldistance = 0.75)
        plt.setp( pie, width=width, edgecolor='white')
        
        probs = [float(x[1]) for x in population_possibility]
        labels = [x[0] for x in population_possibility]
        label_max = labels[probs.index(max(probs))]
        if consistency == 'Consistent':
            labels2 = [x if x == label_max else '' for x in labels]
        else:
            labels2 = [x + '?' if x == label_max else '' for x in labels]
        pie2, _ = ax.pie(probs, radius=1-width, labels=labels2, colors=cin, labeldistance = 0.65)
        plt.setp( pie2, width=width, edgecolor='white')

        plt.text(0, 0, sample, ha='center', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(DIR + '/' + sample + '_DonutPlot.svg')
        plt.close('all')

    inFile.close()
    ouFile.close()

DonutPlot(sys.argv[1])
