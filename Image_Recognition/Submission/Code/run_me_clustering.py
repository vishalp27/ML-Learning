import numpy as np
import matplotlib.pyplot as plt
import util
from sklearn.cluster import KMeans
import math

def computeAIC(var,k,lenc):    # Computes the Akaike Information Criterion(AIC) of a particular cluster model
    return var+2*k*lenc

def computeBIC(var,k,lenc,lend):        # Computes the Bayesian Information Criterion(BIC) of a particular cluster model
    return var+0.5*math.log(lend)*k*lenc 

#Load the data. imgs is an array of shape (N,8,8) where each 8x8 array
#corresponds to an image. imgs_vectors has shape (N,64) where each row
#corresponds to a single-long-vector respresentation of the corresponding image.
img_size = (8,8)
imgs, imgs_vectors  = util.loadDataQ1()
rows, cols = imgs_vectors.shape
numclust=2
minaic=0
inertiarr=np.array([])
datacases=np.array([])
km1=KMeans(n_clusters=numclust,random_state=1).fit(imgs_vectors)
minaic=computeAIC(km1.inertia_,numclust,cols)
for k in range(2,20,1):                   # Calculates optimal number of clusters
    km=KMeans(n_clusters=k,random_state=1).fit(imgs_vectors)
    iner=km.inertia_
    inertiarr=np.append(inertiarr,iner)
    aic=computeAIC(iner,k,cols)
    if aic<minaic:
        minaic=aic
        numclust=k

km2=KMeans(n_clusters=numclust,random_state=1).fit(imgs_vectors)
km2.predict(imgs_vectors)
Zs=KMeans(n_clusters=numclust,random_state=1).fit_predict(imgs_vectors)

# Plots Cluster Examples
for k in np.unique(Zs):
    plt.figure(k)
    if np.sum(Zs==k)>0:
      datacases=np.append(datacases,len(imgs_vectors[Zs == k, :]))
  	  
      util.plot_img_array(imgs_vectors[Zs==k,:], img_size,grey=True)
  	
    plt.suptitle("Cluster Examples %d/%d"%(k,numclust))
    name="../Figures/Cluster_Examples"+str(k)+"by"+str(numclust)+".pdf"
    plt.savefig(name)
    plt.clf()
    plt.cla()
    plt.close()

#plt.show()


# Plots Cluster Centers
plt.figure(1)
util.plot_img_array(km2.cluster_centers_, img_size,grey=True)
plt.suptitle("Cluster Centers")
plt.savefig("../Figures/Cluster_Centers.pdf")
plt.clf()
plt.cla()
plt.close()
#plt.show()


# Plots Inertia vs Number of Clusters

inds=np.arange(2,20,1)
labels=["Inertia"]

#Plot a line graph
plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.plot(inds,inertiarr[0:],'sb-', linewidth=3) #Plot the first series in red with circle marker
#plt.plot(inds,values[1,:],'sb-', linewidth=3) #Plot the first series in blue with square marker

#This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("Inertia") #Y-axis label
plt.xlabel("# Clusters") #X-axis label
plt.title("Inertia vs # Clusters") #Plot title
plt.xlim(1,20) #set x axis range
plt.ylim(3000,8000) #Set yaxis range
plt.legend(labels,loc="best")

#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("../Figures/Cluster_Quality.pdf")
plt.clf()
plt.cla()
plt.close()

# Plots Data Cases Vs Number of Clusters

inds=np.arange(len(datacases))
plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.bar(inds, datacases, align='center') #This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("# Data Cases") #Y-axis label
plt.xlabel("# Clusters") #X-axis label
plt.title("# Data Cases vs # Clusters") #Plot title
plt.xlim(0,len(datacases)) #set x axis range
plt.ylim(0,250) #Set yaxis range

plt.gca().set_xticks(inds) #label locations


#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("../Figures/Cluster_Data_Cases.pdf")
plt.clf()
plt.cla()
plt.close()

# Extra Credit: Using Bayesian Information Criterion(BIC) to determine optimal number of clusters


numclust=2
minbic=0
datacases=np.array([])
km1=KMeans(n_clusters=numclust,random_state=1).fit(imgs_vectors)
minbic=computeBIC(km1.inertia_,numclust,cols,rows)
for k in range(2,20,1):
    km=KMeans(n_clusters=k,random_state=1).fit(imgs_vectors)
    iner=km.inertia_
    bic=computeBIC(iner,k,cols,rows)
    if bic<minbic:
        minbic=bic
        numclust=k

km2=KMeans(n_clusters=numclust,random_state=1).fit(imgs_vectors)
km2.predict(imgs_vectors)
Zs=KMeans(n_clusters=numclust,random_state=1).fit_predict(imgs_vectors)

for k in np.unique(Zs):
    plt.figure(k)
    if np.sum(Zs==k)>0:
      datacases=np.append(datacases,len(imgs_vectors[Zs == k, :]))
  	  
      util.plot_img_array(imgs_vectors[Zs==k,:], img_size,grey=True)
  	
    plt.suptitle("Cluster Examples BIC %d/%d"%(k,numclust))
    name="../Figures/Cluster_Examples_BIC"+str(k)+"by"+str(numclust)+".pdf"
    plt.savefig(name)
    plt.clf()
    plt.cla()
    plt.close()

#plt.show()


plt.figure(1)
util.plot_img_array(km2.cluster_centers_, img_size,grey=True)
plt.suptitle("Cluster Centers BIC")
plt.savefig("../Figures/Cluster_Centers_BIC.pdf")
plt.clf()
plt.cla()
plt.close()
#plt.show()



