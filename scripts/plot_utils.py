from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import operator

def plot_f1(f1s, graph_path, title):
    # f1s is a disc, might contain train/valid/test f1
    plt.clf()
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.ylim([0, 100])
    # print(len(f1s['train-F1']))
    plt.xlim([-1, len(f1s['train'])])
    plt.title(title)
    color_dict = {
        'train': 'b',
        'valid': 'g',
        'test': 'y'
    }
    ordered_f1s = sorted(f1s.items(), key=operator.itemgetter(0))
    
    for label, f1_list in ordered_f1s:
        label_msg = "{:s}: Best ({:.2f}) at epoch {:d}".format(label, max(f1_list), f1_list.index(max(f1_list)))
        plt.plot(f1_list, label=label_msg, color=color_dict[label])
    plt.legend(loc="lower right")
    
    plt.savefig(graph_path, dpi=600, format='svg',
                bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
    plt.close()
