import numpy as np

class Linear():
    def __init__(self, in_features, out_features, init_params=False):

        self.in_features = in_features
        self.out_features = out_features

        self.weight_gradient = 0
        self.bias_gradient = 0

        self.init_params = init_params

        self.weight = np.random.standard_normal(size=(self.out_features, self.in_features))/100
        self.b_rfa = np.random.standard_normal(size=(self.out_features, self.in_features))/100
        self.bias = np.random.standard_normal(size=self.out_features)/100

    def forward(self, x):

        self.input_map = x
        linear_out = np.dot(x, np.transpose(self.weight)) + self.bias
        return linear_out

    # BP
    def calc_gradient(self, error):

        self.error = error
        self.weight_gradient = np.dot(np.transpose(error), self.input_map)
        self.bias_gradient = np.sum(np.transpose(self.error), axis=1)

        next_error = np.dot(error, self.weight)
        return next_error

    # RFA
    def calc_gradient_rfa(self, error):

        self.error = error
        self.weight_gradient = np.dot(np.transpose(error), self.input_map)
        self.bias_gradient = np.sum(np.transpose(self.error), axis=1)

        next_error = np.dot(error, self.b_rfa)
        return next_error

    # DFA
    def calc_gradient_dfa(self, direct_feedback_error):

        self.error = direct_feedback_error
        self.weight_gradient = np.dot(np.transpose(direct_feedback_error), self.input_map)
        self.bias_gradient = np.sum(np.transpose(self.error), axis=1)
        next_error = np.dot(self.error, self.weight)
        return next_error

    # BP
    def backward(self, lr=0.01):
        self.weight -= lr*self.weight_gradient
        self.bias -= lr*self.bias_gradient

        self.weight_gradient = 0
        self.bias_gradient = 0

    # RFA
    def backward_rfa(self, lr=0.001):
        self.weight -= lr*self.weight_gradient
        self.bias -= lr*self.bias_gradient

        self.weight_gradient = 0
        self.bias_gradient = 0

    # DFA
    def backward_dfa(self, lr=0.0001):
        self.weight -= lr*self.weight_gradient
        self.bias -= lr*self.bias_gradient

        self.weight_gradient = 0
        self.bias_gradient = 0

    def save_b_rfa(self, name):
        np.save(name, self.b_rfa)

    def save_weight(self, name):
        np.save(name, self.weight)

    def save_bias(self, name):
        np.save(name, self.bias)

    def load_weight(self, b_rfa, weight, bias):
        self.b_rfa = b_rfa
        self.weight = weight
        self.bias = bias
