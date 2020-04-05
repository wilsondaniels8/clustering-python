#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:13:43 2020

@author: dan
"""
import numpy as np
import math
import copy
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt 
from sklearn import datasets

class Cluster:
    def __init__(self, xpos, ypos, lab):
        self.xcent = xpos
        self.ycent = ypos
        self.data_points = np.empty([0], dtype = object)
        self.xmeanpos = None
        self.ymeanpos = None
        self.xs = np.empty([0], dtype = float)
        self.ys = np.empty([0], dtype = float)
       
        self.label = lab
        
    def clearpoints(self):
        self.data_points = np.empty([0], dtype = object)
        self.xs = np.empty([0], dtype = float)
        self.ys = np.empty([0], dtype = float)
        
    def addpoint(self, point):
        if self.data_points.shape == 0:
            self.data_points = np.array(point)
            self.xs = np.array(point.xval)
            self.ys = np.array(point.yval)
        else:
            self.data_points = np.append(self.data_points, point)
            self.xs = np.append(self.xs, point.xval)
            self.ys = np.append(self.ys, point.yval)
      
    
    def calc_meanpos(self):
        
        self.xmeanpos = np.sum(self.xs)/len(self.xs)
        self.ymeanpos = np.sum(self.ys)/len(self.ys)
        
    def reterror(self):
        kv=0
        totdist = 0
        for i in range(0, self.data_points.shape[0]):
           distx = (self.xs[kv] - self.xcent)**2
           disty = (self.ys[kv] - self.ycent)**2
           kv += 1
           totdist += distx + disty
        return totdist
    
    def move_cluster(self):
        self.xcent = self.xmeanpos
        self.ycent = self.ymeanpos
        
    def members(self):
        for i in self.data_points:
            yield i
            
                
                
class Point:
    def __init__(self, xvalue, yvalue, lab = None):
        self.xval = xvalue
        self.yval = yvalue
        if lab:
            self.label = lab
            
    def set_label(self, lab):
        self.label = lab
        
    def calc_dist(self, another):
        dist = math.sqrt((self.xval - another.xval)**2 + (self.yval - another.yval)**2)
        return dist
    
    
    def closestClust(self, clusters):
        kclust = 0
        closest = -1    #indicates error
        mindist = float("Inf")
        for i in clusters:
            dist = math.sqrt((self.xval - i.xcent)**2 + (self.yval - i.ycent)**2)
            if dist < mindist:
                closest = i
                mindist = dist
            kclust += 1
        return closest
    
        
def makedata():           
    center_1 = np.array([1,1])
    center_2 = np.array([5,5])
    center_3 = np.array([8,1])
    
    # Generate random data and center it to the three centers
    data_1 = np.random.randn(100, 2) + center_1
    data_2 = np.random.randn(100,2) + center_2
    data_3 = np.random.randn(100,2) + center_3
    
    data = np.concatenate((data_1, data_2, data_3), axis = 0)
    
    plt.scatter(data[:,0], data[:,1], s=7)
    df = pd.DataFrame(data = dat, columns = ["x", "y"])
    df.to_csv("points.csv", sep = "\t", index=False)
    
    return data  

def readdata():
     df2 = pd.read_csv("testfile", sep = "\t")
     xvals = np.array(df2['x'])
     yvals = np.array(df2['y'])
     xyval = np.zeros([xvals.shape[0], 2])
     xyval[:,0] = xvals
     xyval[:,1] = yvals
     np.random.shuffle(xyval)
     return xyval
 
def make_coord(clustl):  
    clusters = copy.deepcopy(clustl)
    kv = 0
    clus = np.zeros([len(clustl), 2]) 
    for i in clusters:
        clus[kv, 0] = i.xcent
        clus[kv, 1] = i.ycent
        kv += 1
    return clus

def runcluster(k, dat):
    '''
    runs the algorithm
    '''
    diff = []

    
    # Generate random data and center it to the three centers

    
    #k = 3
    # Number of training data
    n = dat.shape[0]
    # Number of features in the data
    c = dat.shape[1]
    
    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(dat, axis = 0)
    std = np.std(dat, axis = 0)
    xcenters = np.random.uniform(low=np.min(dat[:,0]), high=np.amax(dat[:,0]), size=k)  #fix this
    ycenters = np.random.uniform(low=np.min(dat[:,1]), high=np.amax(dat[:,1]), size=k)  #fix this
    centers = np.zeros([k, 2])
    centers[:,0] = xcenters
    centers[:,1] = ycenters
    
    
    # Plot the data and the centers generated as random
    #plt.scatter(data[:,0], data[:,1], s=7)
    #plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150) 
    
    #initialize data
    pointlist = []
    for i in range(0, dat.shape[0]):
        pointlist.append(Point(dat[i,0], dat[i, 1]))
        
    clustlist = []
    for i in range(0, k):
        clustlist.append(Cluster(centers[i, 0], centers[i, 1], i))

  
    oldclust = copy.deepcopy(clustlist)
  
    kv = 0
    clustcoord = make_coord(clustlist)
    difflist = []
    
    # main loop is here
    
    while True:
        for i in clustlist:
            i.clearpoints()
        for i in pointlist:
            targ = i.closestClust(clustlist)
            #print(targ.label)
            i.set_label(targ.label)
            targ.addpoint(i)
            #print(i.label)
            #print(targ.label)
            
        oldcoord = make_coord(clustlist)
        for i in clustlist:
            if len(i.data_points) == 0:
                return -1, -1
            i.calc_meanpos()
            i.move_cluster()
        newcoord = make_coord(clustlist)
        diff = oldcoord - newcoord
        difflist.append(diff)
        if np.all(diff == 0):
            break
        
    return clustlist, pointlist

        
def plotdata(cl, point):
    '''
    pass in cl clusters and point points objects for plot
    altered color because of strange behavior
    '''
    colors = {0:'blue', 1:'green', 2:'red', 3:'orange', 4:'cyan', 5:'purple'}
    clc = np.zeros([len(cl), 2])
    poi = np.zeros([len(point), 3])
    ccount = 0
    pcount = 0
    
    for i in cl:
        clc[ccount, 0] = i.xcent
        clc[ccount, 1] = i.ycent
        for j in i.members():
            poi[pcount, 0] = j.xval
            poi[pcount, 1] = j.yval
            poi[pcount, 2] = j.label
            pcount += 1
        ccount += 1
    pcolor = np.empty([0], dtype="S10")
    for i in range(poi.shape[0]):
        pcolor = np.append(pcolor, colors[poi[i, 2]])
    print(pcolor)
    plt.scatter(poi[:,0], poi[:,1], s=25, c = pcolor)
    plt.scatter(clc[:,0], clc[:,1], marker='*', c='black', s=150)
    return clc, poi
    plt.rcParams["figure.figsize"] = (12,6)
    #plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])        
        
           
def somedata():
    url = 'https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv'
    # there will be 6 clusters in this data
    df1 = pd.read_csv(url, error_bad_lines=False) 
    row1 = df1.iloc[:,3]
    row1n = row1.to_numpy()
    row2 = df1.iloc[:,4]
    row2n = row2.to_numpy()
    data1 = np.vstack((row1n, row2n))
    data1 = np.transpose(data1)
    return data1
#clist = -1
#while clist == -1:
#    cllist, pointlist = runcluster(5, data1)
def irisdata():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    colors = {0:'blue', 1:'green', 2:'red', 3:'orange', 4:'cyan', 5:'purple'}
    color = [colors[int(y[i])] for i in range(0, y.shape[0])]
    return X

def runsim(clusnum, dat):
    minerr = float("Inf")
    bestclusts = None
    bestpoints = None
    for i in range(0, 30):
        err = 0
        clusters ,pts = runcluster(clusnum, dat)
        if clusters == -1 or pts == -1:   # empty cluster
            continue
        print(clusters, pts)
        for j in clusters:
            err += j.reterror()
        if err < minerr:
            bestclusts = clusters
            bestpoints = pts
            minerr = err
            
            
            
    plotdata(bestclusts, bestpoints)
    return bestpoints, bestclusts
    

       
        
        
        
        
    
         
            
            
        
        
        
    
    
    
    
    

         
        
    
        
        
        