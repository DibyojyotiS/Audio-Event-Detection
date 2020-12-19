import os
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

def single_bar_plot(x, h, plot_dir, title):
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    fig = plt.figure(figsize=[16,5]); ax = fig.add_subplot()
    ax.bar(x, h)
    ax.set_title(title)
    plt.savefig(f"{plot_dir}/{title}.png")
    plt.show()


plot_dir = "cellar/plots"

files_n_classes = np.loadtxt(
    fname= "#shared_train/labels_train.csv",
    delimiter=",",
    dtype='S',
    skiprows=1
).astype(str)

for i in range(len(files_n_classes)):
    files_n_classes[i][0] = files_n_classes[i][0].split('.')[0]

unique_classes, counts = np.unique(files_n_classes[:,1], return_counts=True)

exmpl_per_class = 10
removed_indices = []
dev_indices = []
test_indices = []
for c in unique_classes:
    indices = np.squeeze(np.argwhere(files_n_classes[:,1] == c))
    indices = np.random.choice(indices, size=2*exmpl_per_class, replace=False)
    dev_indices.extend(indices[0:exmpl_per_class])
    test_indices.extend(indices[exmpl_per_class:])
    removed_indices.extend(indices)

dev_files_n_classes = files_n_classes[dev_indices]
test_files_n_classes = files_n_classes[test_indices]
train_files_n_classes = np.delete(files_n_classes, removed_indices, 0)

m,c = np.unique(test_files_n_classes[:,1], return_counts=True)
single_bar_plot(list(m.astype(str)), c, plot_dir, "class-frequencies test-set")
plt.show()

m,c = np.unique(train_files_n_classes[:,1], return_counts=True)
single_bar_plot(list(m.astype(str)), c, plot_dir, "class-frequencies train-set")
plt.show()

m,c = np.unique(dev_files_n_classes[:,1], return_counts=True)
single_bar_plot(list(m.astype(str)), c, plot_dir, "class-frequencies dev-set")
plt.show()

if not os.path.exists("cellar"): os.makedirs("cellar")
np.savetxt("cellar/test_files_n_classes.txt", test_files_n_classes, fmt='%s')
np.savetxt("cellar/dev_files_n_classes.txt", dev_files_n_classes, fmt='%s')
np.savetxt("cellar/train_files_n_classes.txt", train_files_n_classes, fmt='%s')
