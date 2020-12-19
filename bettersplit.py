import os
import numpy as np
import matplotlib.pyplot as plt

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

# [fsID]-[ClassID]-[OccurenceID]-[SliceID]: need to group files from same
fsID_ClassID_group = {}
ClassID_group = {c:[] for c in unique_classes}
for file_name, c in files_n_classes:
    fsID, ClassID, OccurenceID = file_name.split('-')[0:3]
    key = f"{fsID}-{ClassID}-{OccurenceID}"
    if key in fsID_ClassID_group:
        fsID_ClassID_group[key].append(file_name)
    else:
        fsID_ClassID_group[key] = [file_name]
        ClassID_group[c].append(key)

# make the test and dev sets
exmpl_per_class = 10
remove_keys = []
test_keys = {c:[] for c in unique_classes}
dev_keys = {c:[] for c in unique_classes}
for c in unique_classes:
    fsID_ClassIDs = ClassID_group[c]
    nums_files = [len(fsID_ClassID_group[key]) for key in fsID_ClassIDs]
    sorted_idx = np.argsort(nums_files)
    num_files_sorted = np.asarray(nums_files)[sorted_idx]
    num_sorted_keys = np.asarray(fsID_ClassIDs)[sorted_idx] 

    test_exmpls_c = 0
    dev_empls_c = 0
    for key, num in zip(num_sorted_keys, num_files_sorted):
        if dev_empls_c < exmpl_per_class:
            fns = fsID_ClassID_group[key]
            dev_keys[c].append(key)
            dev_empls_c += num
            remove_keys.append(key)
        elif test_exmpls_c < exmpl_per_class:
            test_keys[c].append(key)
            test_exmpls_c += num
            remove_keys.append(key)
        else:
            break

removed_files = []
for key in remove_keys:
    removed_files.extend(fsID_ClassID_group[key])

test_files_n_classes = []
for k in test_keys:
    keys_k = test_keys[k]
    files_k = []
    for key_k in keys_k:
        files_k.extend(fsID_ClassID_group[key_k])
    test_files_n_classes.extend([[file, k] for file in files_k])

dev_files_n_classes = []
for k in dev_keys:
    keys_k = dev_keys[k]
    files_k = []
    for key_k in keys_k:
        files_k.extend(fsID_ClassID_group[key_k])
    dev_files_n_classes.extend([[file, k] for file in files_k])

train_files_n_classes = []
for file, c in files_n_classes:
    if file not in removed_files:
        train_files_n_classes.append([file, c])


# convert to np arrays
test_files_n_classes = np.asarray(test_files_n_classes)
dev_files_n_classes = np.asarray(dev_files_n_classes)
train_files_n_classes = np.asarray(train_files_n_classes)


# plot histograms
m,c = np.unique(test_files_n_classes[:,1], return_counts=True)
single_bar_plot(list(m.astype(str)), c, plot_dir, "class-frequencies test-set")
plt.show()

m,c = np.unique(train_files_n_classes[:,1], return_counts=True)
single_bar_plot(list(m.astype(str)), c, plot_dir, "class-frequencies train-set")
plt.show()

m,c = np.unique(dev_files_n_classes[:,1], return_counts=True)
single_bar_plot(list(m.astype(str)), c, plot_dir, "class-frequencies dev-set")
plt.show()


# save the splits
if not os.path.exists("cellar"): os.makedirs("cellar")
np.savetxt("cellar/test_files_n_classes.txt", test_files_n_classes, fmt='%s')
np.savetxt("cellar/dev_files_n_classes.txt", dev_files_n_classes, fmt='%s')
np.savetxt("cellar/train_files_n_classes.txt", train_files_n_classes, fmt='%s')
