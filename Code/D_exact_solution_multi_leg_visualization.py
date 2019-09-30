"""
This script visualizes a given multi_leg_visualization
"""

from B_helper import *
from mpl_toolkits.mplot3d import Axes3D

#%%
result_folder = "C:\\Users\\Stefan\\LRZ Sync+Share\\Masterarbeit-Klein\\Code\\Results\\exampleStefan-True-ES-MultiLeg-190827-1516"
no_purchase_preference = np.array([2, 2, 1, 2])
capacity_max = np.array([8, 4, 4, 8])
#%%
with open(result_folder+"\\totalresults.data", "rb") as file:
    data_raw = pickle.load(file)

#%%
data_raw = data_raw[str(no_purchase_preference)]
data = pd.DataFrame(data_raw[0]).assign(t=0)

for t in (np.array(range(len(data_raw)-1))+1):
    data = data.append(pd.DataFrame(data_raw[t]).assign(t=t))

data = data.set_index("t", append=True)
for i in np.array(range(len(data.index.names)-1))[::-1]:
    data = data.swaplevel(i+1, i)

data = data["value"]


#%%
def plot_to_file(data, use, c, file_name):
    df = data.loc[use]
    df = df.reset_index()

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_trisurf(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], cmap=plt.cm.viridis)
    ax.set_xlabel("t")
    ax.set_ylabel(c)
    ax.set_zlabel("value")
    ax.set_xticks(np.linspace(0, 20, 5, dtype=int))
    ax.set_yticks(np.arange(max(df.iloc[:, 1])+1))
    fig.savefig(result_folder + "\\" + file_name + ".pdf", bbox_inches="tight")


#%%
idx = pd.IndexSlice
use = idx[:, :, capacity_max[1], capacity_max[2], capacity_max[3]]
file_name = "ES-Value-c1"
c = "c on leg 1"
plot_to_file(data, use, c, file_name)

idx = pd.IndexSlice
use = idx[:, capacity_max[0], :, capacity_max[2], capacity_max[3]]
file_name = "ES-Value-c2"
c = "c on leg 2"
plot_to_file(data, use, c, file_name)

idx = pd.IndexSlice
use = idx[:, capacity_max[0], capacity_max[1], :, capacity_max[3]]
file_name = "ES-Value-c3"
c = "c on leg 3"
plot_to_file(data, use, c, file_name)

idx = pd.IndexSlice
use = idx[:, capacity_max[0], capacity_max[1], capacity_max[2], :]
file_name = "ES-Value-c4"
c = "c on leg 4"
plot_to_file(data, use, c, file_name)