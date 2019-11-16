import numpy as np


class Experiment:
    def __init__(self, experiments):
        self.mean = np.mean(experiments, axis=0)
        self.cov = np.cov(experiments, rowvar=False)