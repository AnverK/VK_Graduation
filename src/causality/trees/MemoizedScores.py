from src.causality.trees.scores import mutual_info


class MemoizedScores:
    def __init__(self, data, score_function=mutual_info):
        self.score_function = score_function
        self.data = data
        self.cache = {}

    def calc_score(self, X, y):
        key = (y, frozenset(X))
        if key in self.cache:
            return self.cache[key]
        value = self.score_function(self.data[:, X], self.data[:, y])
        self.cache[key] = value
        return value
