import numpy as np
from activation import LogSoftmax

class CrossEntropyLoss():
    def __init__(self):
        self.batch_size = 1
        self.target_num = 1

    def forward_loss(self, input, target):
        self.input = input
        self.target = target

        self.logsoftmax_module = LogSoftmax()
        self.logsoftmax_out = self.logsoftmax_module.forward(self.input)
        self.batch_size, self.target_num = self.input.shape

        self.target_one_hot =self.target
        nll_log = -self.logsoftmax_out*self.target_one_hot
        return 1.0/self.batch_size * np.sum(nll_log)

    def calc_gradient_loss(self):
        error1 = -self.target_one_hot
        next_error = self.logsoftmax_module.calc_gradient(error1)
        return 1.0/self.batch_size*next_error


def get_one_hot(targets, nb_classes):
    one_hot = np.zeros((len(targets), nb_classes))
    for i in range(len(targets)):
        one_hot[i, targets[i]] = 1
    return one_hot