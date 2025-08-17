import numpy as np
from sklearn.tree import DecisionTreeRegressor
def get_features_targets(data):
  u_g=data["u"]-data["g"]
  g_r=data["g"]-data["r"]
  r_1=data["r"]-data["i"]
  z=data["i"]-data["z"]
  features=np.vstack((u_g, g_r,r_1,z))
  targets=data["redshift"]
  return np.transpose(features), targets
data = np.load(r"Module 5\5a\sdss_galaxy_colors.npy")
features, targets = get_features_targets(data)
#inicialization
dtr = DecisionTreeRegressor()
#training
dtr.fit(features,targets)
#predictions
predictions=dtr.predict(features)
print(targets[:4])
print(predictions[:4])