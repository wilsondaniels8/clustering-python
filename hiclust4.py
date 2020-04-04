#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:16:16 2020

@author: dan
"""

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
import itertools


#class Link():
#    def __init__(self):
        


class Cluster:
    tot_clusters = 0
    allclusters = []  # this can be examined directly to plot cluster points, simply look for list == length
                      # of desired cluster and plot points
    mergehistory = np.empty([0], dtype = float)
    cluster_history=[]
    def reset_clusters():
        tot_clusters = 0
    def reset_history():
        mergehistory = np.empty([0], dtype = float)
    
    
        
    def __init__(self, xpos, ypos, lab, datap = False):
        self.xcent = xpos
        self.ycent = ypos
        self.data_points = np.empty([0], dtype = object)
        #self.clustlist = np.empty([0], dtype = object)
        self.xmeanpos = None
        self.ymeanpos = None
        self.xs = np.empty([0], dtype = float)
        self.ys = np.empty([0], dtype = float)
        self.typeobj = 'cluster'
        self.num_points = 0
        self.label = lab
        Cluster.tot_clusters += 1
        
        try:
            for i in datap:
                self.addpoints(i)
        except:
           pass
       
    def __str__(self):
        step = "Cluster "+str(self.label) + " with "+str(len(self.data_points))+ " points"
        return step
        
                
        
    def clearpoints(self):
        self.data_points = np.empty([0], dtype = object)
        self.xs = np.empty([0], dtype = float)
        self.ys = np.empty([0], dtype = float)
        self.num_points = 0
        
    
    
        
    def addpoint(self, point):
        if self.data_points.shape == 0:
            self.data_points = np.array(point)
            self.xs = np.array(point.xval)
            self.ys = np.array(point.yval)
            self.num_points = 1
        else:
            self.data_points = np.append(self.data_points, point)
            self.xs = np.append(self.xs, point.xval)
            self.ys = np.append(self.ys, point.yval)
            self.num_points += 1
    
    def merge_clust(self, clus, curr_clusters, dist=0):
        oldclust1 = self.label
        oldclust2 = clus.label
        
        
        num_points_total = self.num_points + clus.num_points
        print("num",num_points_total)
        kv = 0
        
        newcluster = Cluster(0,0, Cluster.tot_clusters)   # increase cluster label # by one using len
        for i in self.data_points:
            newcluster.addpoint(i)
        print("points=",newcluster.data_points)
        for i in clus.data_points:
            newcluster.addpoint(i)
            kv += 1
        
        newcluster.num_points = num_points_total
        kv = 0
        for i in curr_clusters:
            if i.label == self.label:
                ind = kv
                break
            kv +=1
        clustdel = curr_clusters.pop(kv)
        curr_clusters.remove(clus)
        trackval = np.array([oldclust1, oldclust2, dist, num_points_total])
        if Cluster.mergehistory.shape[0] > 0:
            Cluster.mergehistory = np.vstack((Cluster.mergehistory, trackval))
        else:
            Cluster.mergehistory = trackval
        #print(tracklist)
        
        Cluster.most_recent = newcluster
        Cluster.allclusters.append(newcluster)
        return newcluster
    
    def calc_dist(self, clus, method = "average"):
        if method == "average":
            return self.calc_dist_avg(clus)
        elif method == "maximum":
            return self.calc_dist_max(clus)
        elif method == "minimum":
            return self.calc_dist.min(clus)
        
 
    def calc_dist_avg(self, clus):
        maxdist = float("Inf")
        seclust = None
        otclust = None
        dist = 0
        kv = 0
        for i in self.data_points:
            for j in clus.data_points:
                dist=dist + i.calc_dist(j)
                kv += 1
        distav = dist/kv
        return distav
    
    def calc_dist_max(self, clus):
        maxdist = -1
        seclust = None
        otclust = None
        dist = 0
        kv = 0 
        for i in self.data_points:
            for j in clus.data_points:
                dist = i.calc_dist(j)
                kv += 1
                if dist > maxdist:
                    maxdist = dist
        return maxdist
    
    def calc_dist_min(self, clus):
        mindist = float("Inf")
        seclust = None
        otclust = None
        dist = 0
        kv = 0 
        for i in self.data_points:
            for j in clus.data_points:
                dist = i.calc_dist(j)
                kv += 1
                if dist < mindist:
                    mindist = dist
        return mindist
                
                
            
        
        
    
    def calc_meanpos(self):
        
        self.xmeanpos = np.sum(self.xs)/len(self.xs)
        self.ymeanpos = np.sum(self.ys)/len(self.ys)
        
    def printpoints(self):
        k = 0
        print("Cluster "+str(self.label))
        for i in self.data_points:
            print("Point "+str(k)+" "+str(i.xval)+" , "+str(i.yval))
            k += 1
            
    def reterror(self):
        k=0
        totdist = 0
        for i in range(0, data_points.shape):
           distx = (self.xs[k] - self.xcent[k])**2
           disty = (self.ys[k] - self.ycent[k])**2
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
        self.typeobj = "point"
            
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
    df = pd.DataFrame(data = data, columns = ["x", "y"])
    df.to_csv("points.csv", sep = "\t", index=False)
    
    return data  
def readdata():
     df2 = pd.read_csv('/home/dan/Documents/Cluster/hdata.csv' , sep = "\t")
     xvals = np.array(df2['x'])
     yvals = np.array(df2['y'])
     xyval = np.zeros([xvals.shape[0], 2])
     xyval[:,0] = xvals
     xyval[:,1] = yvals
     #np.random.shuffle(xyval)
     return xyval
 
def readdata2():
     df2 = pd.read_csv('/home/dan/Documents/Cluster/hdata.csv' , sep = "\t")
     xvals = np.array(df2['x'])
     yvals = np.array(df2['y'])
     xyval = np.zeros([xvals.shape[0], 2])
     xyval[:,0] = xvals
     xyval[:,1] = yvals
     #np.random.shuffle(xyval)
     return xyval

def makepoints():
    #dat = makedata()
    dat = readdata()
    ptlist = []
    for i in range(dat.shape(0)):
        ptlist.append(Point(dat[i,0]), dat[i, 1], i)
    return ptlist

    
    
def findclose(clusterlist, method="average"):
    mindist = float("Inf")
    clust1 = None
    clust2 = None
    for i, j in itertools.combinations(clusterlist, 2):
        
        dist = i.calc_dist(j, method)
        if dist < mindist:
            clust1 = i
            clust2 = j
        
            mindist = dist
        #print(dist, mindist)
            
    return clust1, clust2, mindist


def mainclust(clusterlist, method="average"):
    for i in clusterlist:
        Cluster.allclusters.append(i)
    while len(clusterlist) > 1:
        clusterold = copy.deepcopy(clusterlist)
        c1, c2, dis = findclose(clusterlist, method)
        anewclust = c1.merge_clust(c2, clusterlist, dis)
        Cluster.cluster_history.append(clusterold)   #keep track of cluster
        clusterlist.append(anewclust)
        #Cluster.allclusters.append(anewclust)
    return Cluster.mergehistory, Cluster.cluster_history


def plotclust(clusthist, numclust):
    # use class variable Cluster.cluster_history as hinput
   colors = {0:"red", 1:"blue", 2:"green", 3:"orange", 4:"purple", 5:"cyan", 6:"yellow"}
   for i in clusthist:
       if len(i) == numclust:
           clusts = i
   num_points = 0 
   for ii in clusts:
       num_points = num_points + len(ii.data_points)
   datmat = np.zeros([num_points, 3])
   dp = 0
   cln = 0
   for ii in clusts:
       for jj in ii.data_points:
           datmat[dp, 0] = jj.xval
           datmat[dp, 1] = jj.yval
           datmat[dp, 2] = cln
           dp += 1
       cln += 1
   col = [colors[int(i)] for i in datmat[:, 2]]    
   plt.scatter(datmat[:,0], datmat[:,1], color = col )
   return datmat

def runhcluster(data, nclus, method = "average"):
    
    aclustl = []
    tl = np.empty([0], dtype = float)

    for i in range(0, data.shape[0]):
        aclustl.append(Cluster(0, 0, Cluster.tot_clusters))  #create a cluster
        aclustl[i].addpoint(Point(dat[i,0], dat[i, 1]))  # put one point into object
    ab, bc = mainclust(aclustl, method)
    plotclust(bc, nclus)
    return ab, bc

def irisdata():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
#dat = dat1
   
#this code just converts data into Point objects and runs the main part of program
#dat = readdata()
#dat = X
#dat = makedata()
            
    




        
     
    
