import numpy as np


class Blackbox:

    def __init__(self, n_sys=20, sd=np.sqrt(10)):

        self.n_sys = n_sys

        self.mu_oracle = [1.5 - 0.5 * i for i in range(self.n_sys)]

        self.sd_oracle = [sd for _ in range(self.n_sys)]

    def sample(self, i):

        return np.random.normal(self.mu_oracle[i], self.sd_oracle[i])


class Blackbox2:

    def __init__(self, n_sys=20, setting='e'):

        self.n_sys = n_sys

        self.mu_oracle = [1.5 - 0.5 * i for i in range(self.n_sys)]

        if setting == 'e':
            self.sd_oracle = [np.sqrt(10) for _ in range(self.n_sys)]
        elif setting == 'i':
            self.sd_oracle = [np.sqrt(10) * (1 + 0.05 * i) for i in range(self.n_sys)]
        else:
            self.sd_oracle = [np.sqrt(10) / (1 + 0.05 * i) for i in range(self.n_sys)]

    def sample(self, i):

        return np.random.normal(self.mu_oracle[i], self.sd_oracle[i])


class Blackbox3:

    def __init__(self, n_sys=20, setting='e'):

        self.n_sys = n_sys

        self.mu_oracle = [1.5 - 0.5 * i for i in range(self.n_sys)]

        if setting == 'e':
            self.sd_oracle = [np.sqrt(10) for _ in range(self.n_sys)]
        elif setting == 'i':
            self.sd_oracle = [np.sqrt(10) * (1 + 0.05 * i) for i in range(self.n_sys)]
        else:
            self.sd_oracle = [np.sqrt(10) / (1 + 0.05 * i) for i in range(self.n_sys)]

    def sample(self, i):

        return np.random.exponential(self.sd_oracle[i]) + self.mu_oracle[i]


if __name__ == '__main__':

    bb = Blackbox3(n_sys=20)

    lst = []

    for _ in range(10000):

        lst.append(bb.sample(1))

    print(np.mean(lst))
    print(np.std(lst))