class PartitionTreeNode(object):
    def __init__(self, left=None, right=None, interval=None):
        self.left = left
        self.right = right
        self.interval = interval

class PartitionTree(object):
    #TODO: This should be a self-balancing tree
    def __init__(self, intervals, labels):
        self.mapping = {}
        self.root = PartitionTreeNode()
        for interval, label in zip(intervals, labels):
            self._add_interval(interval, self.root)
            self.mapping[interval] = label

    def _add_interval(self, interval, starting_node):
        node = starting_node
        left, right = interval
        while node.interval:
            node_left, node_right = node.interval
            if right <= node_left:
                node = node.left
            elif left >= node_right:
                node = node.right
            else:
                raise Exception('Duplicate interval added to tree')
        node.interval = interval
        node.left = PartitionTreeNode()
        node.right = PartitionTreeNode()

    def get_label(self, number):
        interval = self._get_interval(number, self.root)
        return self.mapping[interval]

    def _get_interval(self, number, node):
        left_bound, right_bound = node.interval
        while number < left_bound or number > right_bound:
            if number < left_bound:
                node = node.left
            elif number > right_bound:
                node = node.right
            left_bound, right_bound = node.interval
        return node.interval