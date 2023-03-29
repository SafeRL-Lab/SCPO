import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

sys.path.append(os.path.dirname(__file__) + os.sep + './')


def load_json(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


def smoothen(data, w):
    res = np.zeros_like(data)
    for i in range(len(data)):
        if i > w:
            res[i] = np.mean(data[i-w:i])
        elif i > 0:
            res[i] = np.mean(data[:i])
        else:  # i == 0
            res[i] = data[i]
    return res



def draw(data_dict, env_id, attribute: str, w, i=0):
    color = ['orange', 'hotpink', 'dodgerblue', 'mediumpurple', 'c', 'cadetblue', 'steelblue', 'mediumslateblue',
             'hotpink', 'mediumturquoise']

    """
    color = [
        'aqua',
        'aquamarine',
        'bisque',
        'black',
        'blue',
        'blueviolet',
        'brown',
        'burlywood',
        'cadetblue',
        'chartreuse',
        'chocolate',
        'coral',
        'cornflowerblue',
        'crimson',
        'cyan',
        'darkblue',
        'darkcyan',
        'darkgoldenrod',
        'darkgray',
        'darkgreen',
        'darkkhaki',
        'darkmagenta',
        'darkolivegreen',
        'darkorange',
        'darkorchid',
        'darkred',
        'darksalmon',
        'darkseagreen',
        'darkslateblue',
        'darkslategray',
        'darkturquoise',
        'darkviolet',
        'deeppink',
        'deepskyblue',
        'dimgray',
        'dodgerblue',
        'firebrick',
        'floralwhite',
        'forestgreen',
        'fuchsia',
        'gainsboro',
        'ghostwhite',
        'gold',
        'goldenrod',
        'gray',
        'green',
        'greenyellow',
        'honeydew',
        'hotpink',
        'indianred',
        'indigo',
        'ivory',
        'khaki',
        'lavender',
        'lavenderblush',
        'lawngreen',
        'lemonchiffon',
        'lightblue',
        'lightcoral',
        'lightcyan',
        'lightgoldenrodyellow',
        'lightgreen',
        'lightgray',
        'lightpink',
        'lightsalmon',
        'lightseagreen',
        'lightskyblue',
        'lightslategray',
        'lightsteelblue',
        'lightyellow',
        'lime',
        'limegreen',
        'linen',
        'magenta',
        'maroon',
        'mediumaquamarine',
        'mediumblue',
        'mediumorchid',
        'mediumpurple',
        'mediumseagreen',
        'mediumslateblue',
        'mediumspringgreen',
        'mediumturquoise',
        'mediumvioletred',
        'midnightblue',
        'mintcream',
        'mistyrose',
        'moccasin',
        'navajowhite',
        'navy',
        'oldlace',
        'olive',
        'olivedrab',
        'orange',
        'orangered',
        'orchid',
        'palegoldenrod',
        'palegreen',
        'paleturquoise',
        'palevioletred',
        'papayawhip',
        'peachpuff',
        'peru',
        'pink',
        'plum',
        'powderblue',
        'purple',
        'red',
        'rosybrown',
        'royalblue',
        'saddlebrown',
        'salmon',
        'sandybrown',
        'seagreen',
        'seashell',
        'sienna',
        'silver',
        'skyblue',
        'slateblue',
        'slategray',
        'snow',
        'springgreen',
        'steelblue',
        'tan',
        'teal',
        'thistle',
        'tomato',
        'turquoise',
        'violet',
        'wheat',
        'white',
        'whitesmoke',
        'yellow',
        'yellowgreen']
        """
    plt.xlabel("Environment steps", fontsize=18)
    plt.ylabel(f"Average Episode {attribute}", fontsize=18) # Reward Cost
    for algorithm, tb_data in data_dict[env_id].items():
        attribute_mean = tb_data["data"][attribute+"_mean"]
        attribute_std = tb_data["data"][attribute+"_std"]
        attribute_mean = smoothen(attribute_mean, w)
        attribute_std = smoothen(attribute_std, w)
        t = np.arange(attribute_mean.shape[0])
        plt.plot(t, attribute_mean, color=color[i], label=algorithm, linewidth=1.5)
        #plt.fill_between(t, attribute_mean - attribute_std, attribute_mean + attribute_std, alpha=0.05, color=color[i])
        i += 1
        print(attribute_mean[-1])

data = load_json("/home/jaafar/Documents/safe_rl/plots/pybullet_experiments_combined.json")

if __name__ == "__main__":
    # scenario = "Ant"
    # config = "2x4"
    scenario = "ManyAgent Ant"
    config = "6x1"
    performance_type = "costs" #"costs" # "rewards"
    smoothen_w = 5
    need_legend = True  # False True
    # need_legend = True
    draw(data, "SafetyBallCircle-v0", "return", smoothen_w)
    plt.legend(loc="upper left", fontsize=14)
    plt.show()
