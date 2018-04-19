import util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

patch_size = (10,10)
data_noisy, data_clean, img_noisy, img_clean = util.loadDataQ2(patch_size)
m_noise=10
opt_comp=1
maearr=np.array([])
for comp in range(5,30,1):                                  # Calculates optimal number of components
    pca=PCA(n_components=comp)
    trans=pca.fit_transform(data_noisy)
    data_denoised=pca.inverse_transform(trans)
    img_denoised=util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))
    mae=util.eval_recon(img_clean,img_denoised)
    maearr=np.append(maearr,mae)
    if mae<m_noise:
        m_noise=mae
        opt_comp=comp


pca=PCA(n_components=opt_comp)
trans=pca.fit_transform(data_noisy)
data_denoised=pca.inverse_transform(trans)
img_denoised=util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))

#Plot the clean and noisy images
plt.figure(0,figsize=(7,3))
util.plot_pair(img_clean,img_noisy,"Clean","Noisy")
plt.savefig("../Figures/Clean_Vs_Noisy_Image.pdf")
plt.clf()
plt.cla()
plt.close()

#Plot the clean and de-noised images
plt.figure(1,figsize=(7,3))
util.plot_pair(img_clean,img_denoised,"Clean","De-Noised")
plt.savefig("../Figures/Clean_Vs_Denoised_Image.pdf")
plt.clf()
plt.cla()
plt.close()

inds=np.arange(5,30,1)
labels=["MAE"]

# Plot MAE Vs Number of Components

#Plot a line graph
plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.plot(inds,maearr[0:],'sb-', linewidth=3) #Plot the first series in red with circle marker
#plt.plot(inds,values[1,:],'sb-', linewidth=3) #Plot the first series in blue with square marker

#This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("MAE") #Y-axis label
plt.xlabel("# Components") #X-axis label
plt.title("MAE vs # Components-PCA") #Plot title
plt.xlim(1,30) #set x axis range
plt.ylim(0.05,0.06) #Set yaxis range
plt.legend(labels,loc="best")

#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("../Figures/MAE_Vs_Components.pdf")
plt.clf()
plt.cla()
plt.close()


# EXTRA CREDIT: USING Non-Negative Matrix Factorization(NMF)


m_noise=10
opt_comp=1
maearr=np.array([])
for comp in range(5,30,1):                        # Calculates optimal number of components
    pca=NMF(n_components=comp)
    trans=pca.fit_transform(data_noisy)
    data_denoised=pca.inverse_transform(trans)
    img_denoised=util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))
    mae=util.eval_recon(img_clean,img_denoised)
    maearr=np.append(maearr,mae)
    if mae<m_noise:
        m_noise=mae
        opt_comp=comp


pca=NMF(n_components=opt_comp)
trans=pca.fit_transform(data_noisy)
data_denoised=pca.inverse_transform(trans)
img_denoised=util.patchToImage(util.vectorsToPatches(data_denoised,patch_size),(900,1200))

#Plot the clean and de-noised images
plt.figure(1,figsize=(7,3))
util.plot_pair(img_clean,img_denoised,"Clean","De-Noised")
plt.savefig("../Figures/Clean_Vs_Denoised_Image_NMF.pdf")
plt.clf()
plt.cla()
plt.close()

inds=np.arange(5,30,1)
labels=["MAE"]

# Plot MAE Vs Number of Components

#Plot a line graph
plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.plot(inds,maearr[0:],'sb-', linewidth=3) #Plot the first series in red with circle marker
#plt.plot(inds,values[1,:],'sb-', linewidth=3) #Plot the first series in blue with square marker

#This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("MAE") #Y-axis label
plt.xlabel("# Components") #X-axis label
plt.title("MAE vs # Components-NMF") #Plot title
plt.xlim(1,30) #set x axis range
plt.ylim(0.05,0.06) #Set yaxis range
plt.legend(labels,loc="best")

#Make sure labels and titles are inside plot area
plt.tight_layout()

#Save the chart
plt.savefig("../Figures/MAE_Vs_Components_NMF.pdf")
plt.clf()
plt.cla()
plt.close()