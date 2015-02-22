class PartitionTreeNode(object):
    def __init__(self, left=None, right=None, interval=None):
        self.left = left
        self.right = right
        self.interval = interval

class PartitionTree(object):
    def __init__(self, intervals, labels):
        self.mapping = {}
        self.root = PartitionTreeNode()
        for interval, label in zip(intervals, labels):
            self._add_interval(interval, self.root)
            self.mapping[interval] = label

    def _add_interval(self, interval, node):
        if not node.interval:
            node.interval = interval
            node.left = PartitionTreeNode()
            node.right = PartitionTreeNode()
        elif interval[1] <= node.interval[0]:
                self._add_interval(interval, node.left)
        elif interval[0] >= node.interval[1]:
                self._add_interval(interval, node.right)
        else:
            raise Exception

    def get_label(self, number):
        interval = self._get_interval(number, self.root)
        return self.mapping[interval]

    def _get_interval(self, number, node):
        left_bound, right_bound = node.interval
        if number < left_bound:
            return self._get_interval(number, node.left)
        elif number > right_bound:
            return self._get_interval(number, node.right)
        else:
            return node.interval