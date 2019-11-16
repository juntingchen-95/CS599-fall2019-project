from copy import copy
import numpy as np
from numpy import ndarray


class Node:
    # Build tree leaves.

    attr_names = ("avg", "left", "right", "feature", "split", "mse")

    def __init__(self, avg=None, left=None, right=None, feature=None, split=None, mse=None):
        self.avg = avg
        self.left = left
        self.right = right
        self.feature = feature
        self.split = split
        self.mse = mse

    def __str__(self):
        ret = []
        for attr_name in self.attr_names:
            attr = getattr(self, attr_name)
            # Describe the attribute of Node.
            if attr is None:
                continue
            if isinstance(attr, Node):
                des = "%s: Node object." % attr_name
            else:
                des = "%s: %s" % (attr_name, attr)
            ret.append(des)

        return "\n".join(ret) + "\n"

    def copy(self, node):
        # Copy the attributes of another Node.

        for attr_name in self.attr_names:
            attr = getattr(node, attr_name)
            setattr(self, attr_name, attr)


class RegressionTree:

    def __init__(self):
        self.root = Node()
        self.depth = 1
        self._rules = None

    def __str__(self):
        ret = []
        for i, rule in enumerate(self._rules):
            literals, avg = rule

            ret.append("Rule %d: " % i + ' | '.join(
                literals) + ' => y_hat %.4f' % avg)
        return "\n".join(ret)

    @staticmethod
    def _expr2literal(expr: list) -> str:

        feature, operation, split = expr
        operation = ">=" if operation == 1 else "<"
        return "Feature%d %s %.4f" % (feature, operation, split)

    def get_rules(self):
        # Get the rules of all the tree nodes.

        # Breadth-First Search.
        que = [[self.root, []]]
        self._rules = []

        while que:
            node, exprs = que.pop(0)

            # Generate a rule when the current node is leaf node.
            if not(node.left or node.right):
                # Convert expression to text.
                literals = list(map(self._expr2literal, exprs))
                self._rules.append([literals, node.avg])

            # Expand when the current node has left child.
            if node.left:
                rule_left = copy(exprs)
                rule_left.append([node.feature, -1, node.split])
                que.append([node.left, rule_left])

            # Expand when the current node has right child.
            if node.right:
                rule_right = copy(exprs)
                rule_right.append([node.feature, 1, node.split])
                que.append([node.right, rule_right])

    @staticmethod
    def _get_split_mse(col: ndarray, label: ndarray, split: float) -> Node:
        # Calculate the mse of label when col is splitted into two pieces.

        # Split label.
        label_left = label[col < split]
        label_right = label[col >= split]

        # Calculate the means of label.
        avg_left = label_left.mean()
        avg_right = label_right.mean()

        # Calculate the mse of label.
        mse = (((label_left - avg_left) ** 2).sum() +
               ((label_right - avg_right) ** 2).sum()) / len(label)

        # Create nodes to store result.
        node = Node(split=split, mse=mse)
        node.left = Node(avg_left)
        node.right = Node(avg_right)

        return node

    def _choose_split(self, col: ndarray, label: ndarray) -> Node:

        # Feature cannot be splitted if there's only one unique element.
        node = Node()
        unique = set(col)
        if len(unique) == 1:
            return node

        # In case of empty split.
        unique.remove(min(unique))

        # Get split point which has min mse.
        ite = map(lambda x: self._get_split_mse(col, label, x), unique)
        node = min(ite, key=lambda x: x.mse)

        return node

    def _choose_feature(self, data: ndarray, label: ndarray) -> Node:

        # Compare the mse of each feature and choose best one.
        _ite = map(lambda x: (self._choose_split(data[:, x], label), x),
                   range(data.shape[1]))
        ite = filter(lambda x: x[0].split is not None, _ite)

        # Return None if no feature can be splitted.
        node, feature = min(
            ite, key=lambda x: x[0].mse, default=(Node(), None))
        node.feature = feature

        return node

    def fit(self, data: ndarray, label: ndarray, max_depth=5, min_samples_split=2):

        # Initialize with depth, node, indexes.
        self.root.avg = label.mean()
        que = [(self.depth + 1, self.root, data, label)]

        # Breadth-First Search.
        while que:
            depth, node, _data, _label = que.pop(0)

            # Terminate loop if tree depth is more than max_depth.
            if depth > max_depth:
                depth -= 1
                break

            # Stop split when number of node samples is less than
            # min_samples_split or Node is 100% pure.
            if len(_label) < min_samples_split or all(_label == label[0]):
                continue

            # Stop split if no feature has more than 2 unique elements.
            _node = self._choose_feature(_data, _label)
            if _node.split is None:
                continue

            # Copy the attributes of _node to node.
            node.copy(_node)

            # Put children of current node in que.
            idx_left = (_data[:, node.feature] < node.split)
            idx_right = (_data[:, node.feature] >= node.split)
            que.append(
                (depth + 1, node.left, _data[idx_left], _label[idx_left]))
            que.append(
                (depth + 1, node.right, _data[idx_right], _label[idx_right]))

        # Update tree depth and rules.
        self.depth = depth
        self.get_rules()

    def predict_one(self, row: ndarray) -> float:
        # Auxiliary function of predict.

        node = self.root
        while node.left and node.right:
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right

        return node.avg

    def predict(self, data: ndarray) -> ndarray:

        return np.apply_along_axis(self.predict_one, 1, data)


def main():
    # Load data
    data, label = load_data()
    # Split data randomly, train set rate 70%
    data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=200)
    # Train model
    reg = RegressionTree()
    reg.fit(data=data_train, label=label_train, max_depth=5)
    # Show rules
    print(reg)


if __name__ == "__main__":
    main()
