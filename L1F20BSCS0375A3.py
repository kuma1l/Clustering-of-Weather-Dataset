import numpy as np
import pandas as df
import statistics
import matplotlib.pyplot as plt
import sys


def load_and_preprocess():
    data=df.read_csv('minute_weather.csv')
    data.dropna(inplace=True)
    features = ['rowID','air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction',
                    'max_wind_speed', 'relative_humidity']
    data=data[features]
    data=(np.array(data).astype(float))
    row=0
    for i in range(len(data)):
        data[i][0]=row
        row+=1
    return data

def only_csv():
    data=df.read_csv('minute_weather.csv')
    data.dropna(inplace=True)
    features = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction',
                    'max_wind_speed', 'relative_humidity']
    data=data[features]
    
    return data
    

def distance(l1, l2):
    summ=0
    for i in range(1,len(l1)):
        summ+=(l1[i]-l2[i])**2
    
    return summ**0.5
        


def k_means(data):
    centroid=[0,124,8000,10000]
    newcent=[0]*4
    cents=[]
    
    cluster={}
    
    cluster=dict(cluster)
    
    for i in centroid:
        cents.append(data[i])
        
        
    iterr=0
        
    while(newcent != centroid):
    
        newcent[0]=centroid[0]
        newcent[1]=centroid[1]
        newcent[2]=centroid[2]
        newcent[3]=centroid[3]
        
        cluster[0]=[]
        cluster[1]=[]
        cluster[2]=[]
        cluster[3]=[]
        
        
        for i in range(len(data)):
            if i in centroid:
                continue
            else:
                d1=distance(data[i], cents[0])
                d2=distance(data[i], cents[1])
                d3=distance(data[i], cents[2])
                d4=distance(data[i], cents[3])
            
                minn=min(d1,d2,d3,d4)
            
                if minn==d1:
                    cluster[0].append(i)
                elif minn==d2:
                    cluster[1].append(i)
                elif minn==d3:
                    cluster[2].append(i)
                elif minn==d4:
                    cluster[3].append(i)
                    
        centroid[0]=int(statistics.fmean(cluster[0]))
        centroid[1]=int(statistics.fmean(cluster[1]))
        centroid[2]=int(statistics.fmean(cluster[2]))
        centroid[3]=int(statistics.fmean(cluster[3]))
        
        cents=[]
        for i in centroid:
            cents.append(data[i])
        print('Iteration:',iterr)
        print(centroid)
        print(newcent)
        
        if iterr==12:
            break
        iterr+=1
    return cluster


def plus_plus(ds, k, random_state=42):
    np.random.seed(random_state)
    centroids = [ds[0]]
    retcent=[0]

    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        centroids.append(ds[i])
        retcent.append(i)

    return retcent

def kmeans_plus_plus(data, centroid, k):
    newcent=[0]*k
    cents=[]
    iterr=0
    cluster={}
    cluster=dict(cluster)
    
    for i in centroid:
        cents.append(data[i])
    while(newcent != centroid):
        
        
        for i in range(k):
            newcent[i]=centroid[i]
            cluster[i]=[]
        
        for i in range(len(data)):
            if i in centroid:
                continue
            else:
                dArr=[]
                for j in range(k):
                    dArr.append(distance(data[i], cents[j]))

                minn=min(dArr)
                
                for j in range (k):
                    if minn==dArr[j]:
                        cluster[j].append(i)
                        
        for j in range(k):
            if len(cluster[j])==0:
                centroid[j]=0
            else:
                centroid[j]=int(statistics.fmean(cluster[j]))
        cents=[]
        for i in centroid:
            cents.append(data[i])
        print('Iteration:',iterr)
        print(centroid)
        print(newcent)
        
        if iterr==12:
            break
        iterr+=1
    return cluster

def intra_Cluster(clusters, k):
    d=0
    for i in range(k):
        for j in range(len(clusters[i])-1):
            for g in range(j+1, len(clusters[i])):
                dist=distance(data[j], data[g])
                if(dist>d):
                    d=dist
    return d

def inter_Cluster(centroid, k):
    d=0.0
    if k==1:
        return 100000
    else:
        d=sys.float_info.max
        for i in range (len(centroid)-1):
            for j in range(i+1, len(centroid)):
                dist=distance(data[i], data[j])
                if dist<d:
                    d=dist
        return d

def dunn_Index(clusters, centroid, k):
    return inter_Cluster(centroid, k)/intra_Cluster(clusters, k)

def optimal_k(data):
    dunn_Arr=[]
    for k in range (2,51):
        centroid=plus_plus(data, k)
        clusters=kmeans_plus_plus(data, centroid, k)
        dunn_Arr.append(dunn_Index(clusters, centroid, k))
    return dunn_Arr

data=load_and_preprocess()

'''cl1=k_means(data)
centroid=plus_plus(data, 5)
cl2=kmeans_plus_plus(data, centroid, 5)'''

dunn=optimal_k(data[:1000])
plt.plot(range(len(dunn)), dunn, marker='o', linestyle='-')
plt.xlabel('Clusters')
plt.ylabel('Dunn Index')
plt.title('Elbow point')
plt.grid(True)
plt.show()












