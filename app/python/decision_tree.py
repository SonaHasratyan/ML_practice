import pandas as pd
import argparse
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--target", type=str, required=True)

parser.add_argument("--max_depth", default=3, type=int, required=False)
parser.add_argument("--criterion", default="gini", type=str, required=False)
parser.add_argument("--max_leaf_nodes", default=5, type=int, required=False)
parser.add_argument("--random_state", default=78, type=int, required=False)

args = parser.parse_args()

csv_dir = args.data_path

df = pd.read_csv(csv_dir)
X = df.select_dtypes(include=["number"])
X = X.drop(args.target, axis=1)

y = df[args.target]

dt = DecisionTreeClassifier(
    max_depth=args.max_depth,
    criterion=args.criterion,
    max_leaf_nodes=args.max_leaf_nodes,
    random_state=args.random_state,
)
dt.fit(X, y)

plt.figure(figsize=(20, 16))
plot_tree(dt, filled=True)
plt.savefig(os.path.join(os.getcwd(), "../tree.png"))

plt.show()

