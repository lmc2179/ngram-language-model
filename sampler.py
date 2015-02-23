import random
import partition_tree

class Multinomial_Sampler(object):
    def __init__(self, probabilities, event_names):
        intervals = self._build_intervals_from_probabilities(probabilities)
        self.tree = partition_tree.PartitionTree(intervals, event_names)

    def _build_intervals_from_probabilities(self, probabilities):
        intervals = []
        left_side = 0.0
        for p in probabilities:
            intervals.append((left_side, left_side+p))
            left_side += p
        return intervals


    def sample(self):
        random_0_1 = random.random()
        return self.tree.get_label(random_0_1)
