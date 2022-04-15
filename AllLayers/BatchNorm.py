from AllLayers import Layers
import numpy as np


class BatchNorm(Layers.Layers):
    def __init__(self, input_dim, eps, mom):
        ran = np.sqrt(1 / input_dim)
        self.X_hat = None
        # [gamma, beta]
        self.parameters = [np.ones(input_dim), np.zeros(input_dim)]
        # [dgamma, dbeta]
        self.grads = [np.zeros_like(self.parameters[0]), np.zeros_like(self.parameters[1])]
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.zeros(input_dim)
        self.batch_mean = None
        self.batch_var = None
        self.eps = eps
        self.m = input_dim
        self.mom = mom

    def forward(self, x, mode):
        N, D = x.shape

        if mode == 'train':
            self.batch_mean = np.mean(x, axis=0)
            self.batch_var = np.var(x, axis=0)
            self.X_hat = (x - self.batch_mean.T) / np.sqrt(self.batch_var.T + self.eps)

            self.running_mean = self.mom * self.running_mean + (1 - self.mom) * self.batch_mean
            self.running_var = self.mom * self.running_var + (1 - self.mom) * self.batch_var

            out = self.X_hat * self.parameters[0] + self.parameters[1]
            return out
        elif mode == 'test':
            x_cap = (x - self.running_mean) / np.sqrt(self.running_var)
            return ((x_cap * self.parameters[0]) + self.parameters[1])

    def backward(self, grad, original_input):
        x = original_input
        xcap_hat = grad * self.parameters[0]
        batch_var_hat = np.sum(xcap_hat * (x - self.batch_mean) * (-0.5) * (self.batch_var + self.eps) ** (-1.5),
                               axis=0)
        batch_mean_hat = batch_var_hat * (np.sum((-2 * (x - self.batch_mean))) / self.m) + np.sum(
            xcap_hat * (-1 / np.sqrt(self.batch_var + self.eps)), axis=0)
        x_hat = (xcap_hat * (1 / np.sqrt(self.batch_var + self.eps))) + (batch_mean_hat / self.m) + (
                    batch_var_hat * ((2 * (x - self.batch_mean)) / self.m))
        gamma_hat = np.sum(grad * self.X_hat, axis=0)
        beta_hat = np.sum(grad, axis=0)

        self.grads[0] = gamma_hat
        self.grads[1] = beta_hat
        grad_x = x_hat

        # return gradient with respect to the original input to this layer
        return grad_x

    def __call__(self, x, mode):
        return self.forward(x, mode)