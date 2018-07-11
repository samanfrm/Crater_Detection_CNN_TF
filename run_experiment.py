import time
from datetime import timedelta
from crater_loader import load_crater_data_wrapper
from crater_network import Network
import numpy as np
import matplotlib.pyplot as plt

iteration = 0
experiment_data = []
for _ in range(10):
    iteration += 1

    start = time.time()
    
    # load the training and test data
    tr_d, te_d = load_crater_data_wrapper()
    
    # define the network shape to be used and the activation threshold
    model = Network([40000, 8, 1], False)
    model.threshold = 0.3

    # the schedule is how the learning rate will be
    # changed during the training
    epochs = 100
    schedule = [(0.1)*(0.5)**np.floor(float(i)/(30)) for i in range(epochs)]
    for eta in schedule:
        # the total epochs is given by the schedule loop
        # we chose minibatch size to be 3
        model.SGD(tr_d, 1, 3, eta, te_d)

    end = time.time()

    # After training is complete, store this model training history
    # to the experiment data
    experiment_data.append(np.array(model.history))
    
    # store current results data to disk
    np.save("experiment_data", experiment_data)
    
    elapsed_time = end - start
    print iteration, timedelta(seconds=elapsed_time)


# data numpy array
# axis0: iterations, axis1: epochs, axis2: measurement
data = np.load("experiment_data.npy")

# extract temporal axis
epoch = data[ : , : , 0].mean(axis=0)

# list to store statistics for measurements
stats = []
# helper dictionary
stats_attr = {0: "TP", 1: "FP",  2: "FN",  3: "Detection rate",
              4: "False rate", 5: "Quality rate", 6: "Accuracy"}

for i in range(7):
    # a list for each measurement
    stats.append([])
    # compute statistics along iterations axis
    stats[i].append(data[ : , : , i+1].mean(axis=0))
    stats[i].append(data[ : , : , i+1].std(axis=0))
    stats[i].append(data[ : , : , i+1].min(axis=0))
    stats[i].append(data[ : , : , i+1].max(axis=0))

# finally plot statistics of evolution of our Network performance
plotstats = [3, 4, 5, 6]
fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
fig.tight_layout(h_pad=8, w_pad=4)
fig.subplots_adjust(bottom=0.12, top=0.88, left=0.1)
for k, ax in enumerate(axarr.flatten()):
    i = plotstats[k]
    ax.set_xlabel("Epoch")
    ax.set_ylabel(stats_attr[i])
    ax.plot(epoch, stats[i][0])
    ax.set_title("%s\nLast - mean: %.4f, std: %.4f "
              % (stats_attr[i], stats[i][0][-1], stats[i][1][-1]))
    ax.fill_between(epoch, stats[i][0] - stats[i][1],
                    stats[i][0] + stats[i][1], alpha=0.4)
    ax.fill_between(epoch, stats[i][2], stats[i][3], alpha=0.4)

plt.show()

# Now wee look for some of the missclassified pictures
# they will be shown as plots
wrongpred = []
for _ in range(3):
    # load the training and test data
    tr_d, te_d = load_crater_data_wrapper()

    # define the network shape to be used and the activation threshold
    model = Network([40000, 8, 1], False)
    model.threshold = 0.3

    # the schedule is how the learning rate will be
    # changed during the training
    epochs = 100
    schedule = [(0.1)*(0.5)**np.floor(float(i)/(30)) for i in range(epochs)]
    for eta in schedule:
        # the total epochs is given by the schedule loop
        # we chose minibatch size to be 3
        model.SGD(tr_d, 1, 3, eta, te_d)

    for i, sample in enumerate(te_d):
        pred, y = int(model.feedforward(sample[0])[0][0]>model.threshold), sample[1]
        if not pred == y:
            wrongpred.append(sample)

wrongset = set()
for wp in wrongpred:
    wrongset.add((tuple(list((wp[0]*255).T.astype(np.int)[0])), wp[1]))

missclassified = list(wrongset)

def show_sample(data):
    img = np.array(data[0]).reshape((200,200))
    plt.title("label: %d" % data[1])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.show()

for sample in missclassified:
    show_sample(sample)
