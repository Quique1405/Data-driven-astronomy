import numpy as np
import matplotlib.pyplot as plt
data= np.load(r"Module 5\5a\sdss_galaxy_colors.npy")
def get_features_targets(data):
    u_g = data['u']-data['g']
    g_r = data['g']-data['r']
    r_i = data['r']-data['i']
    features = np.vstack((u_g, g_r, r_i))
    targets = data['redshift']
    return features.transpose(), targets
colour_indexes,redshifts=get_features_targets(data)
colour_indexes = colour_indexes[:,::2]
norm = plt.Normalize(vmin=0, vmax=redshifts[np.argmax(redshifts)]/2)
plt.scatter(colour_indexes[:,0],colour_indexes[:,1],lw=0,s=2.5,c=redshifts,cmap="YlOrRd",norm=norm)
plt.colorbar(label='Redshift')
plt.xlim(-0.5,2.5)
plt.ylim(-0.4,1)
plt.xlabel("Colour index u-g")
plt.ylabel("Colour index i-r")
plt.title("Redshift (colour) u-g vs r-i")
plt.show()