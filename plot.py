### file adapted from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw2

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os


"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L data.csv
        L config.yaml
      L  1
        L data.csv
        L config.yaml
       .
       .
       .
      L  9
        L data.csv
        L config.yaml

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name=='mean_ep_reward':
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def plot_data(data, title="", value="AverageReturn", xmax=60000, ymax=10):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="white", font_scale=1.5)
    sns.lineplot(data=data, y=value, x="step", hue="Condition", linewidth=3.0)#, palette=['C6', 'C2', 'C0', 'C8', 'C3']) #, units="Unit", estimator=None)
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    leg = plt.legend(loc='best', prop={'size': 30}).set_draggable(True)
    # set the linewidth of each legend object
    for legobj in leg.legend.legendHandles:
        legobj.set_linewidth(5.0)
    plt.xlabel("Training steps", fontsize=25)
    plt.ylabel("Average episode success", fontsize=25)
    plt.title(title, fontsize=35)
    plt.show()


def get_datasets(fpath, value, condition=None):

    def load_config(file):
        if file is not None:
            with open(file) as f:
                return yaml.load(f, Loader=yaml.UnsafeLoader)
        return None
    
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'data.csv' in files:
            params = load_config(os.path.join(root,'config.yaml'))
            exp_name = params['general']['exp_name'] if params else 'exp1'
            
            log_path = os.path.join(root,'data.csv')
            experiment_data = pd.read_csv(log_path)
            #experiment_data = normalize(experiment_data)
            
            smooth = False
            if smooth:
              import numpy as np
              col = value
              window_width = 50
              data = experiment_data[col].tolist()
              cumsum_vec = np.cumsum([0]*window_width + data)
              ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
              experiment_data[col] = ma_vec

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )        
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition or exp_name
                )

            datasets.append(experiment_data)
            unit += 1

    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--title', default=[''], nargs='*')
    parser.add_argument('--value', default=['train_envs/success/mean/100'], nargs='*')
    parser.add_argument('--xmax', default=[60000], nargs='*')
    parser.add_argument('--ymax', default=[1], nargs='*')
    args = parser.parse_args()

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            for subdir in next(os.walk(logdir))[1]:
                data += get_datasets(os.path.join(logdir, subdir), args.value, legend_title)
    else:
        for logdir in args.logdir:
            for subdir in next(os.walk(logdir))[1]:
                data += get_datasets(os.path.join(logdir, subdir), args.value)

    for value, title, xmax, ymax in zip(args.value, args.title, args.xmax, args.ymax):
        plot_data(data, title=title, value=value, xmax=float(xmax), ymax=float(ymax))


if __name__ == "__main__":
    main()
