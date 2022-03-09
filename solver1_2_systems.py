from collections import Counter
import numpy as np
import tqdm

"""
numerical study 1
"""


class Blackbox2Sys:

    def __init__(self, diff=1, sd=1):
        self.mu = [0, diff]
        self.sd = sd
        self.n_sys = 2

    def sample(self, i):
        return np.random.normal(self.mu[i], self.sd)


class Procedure2Sys:

    def __init__(self, blackbox, alpha=0.05, delta=1, n0=2):

        self.blackbox = blackbox

        self.alpha = alpha
        self.delta = delta
        self.n0 = n0

        self.sample_count = [0] * 2
        self.sample_total = [0] * 2
        self.sample_sqtot = [0] * 2

        self.mu_hat_prev = [0] * 2
        self.mu_hat_curr = [0] * 2

        self.pi = 0
        self.ll = [0] * 2
        self.glr = [0] * 2

        self.initialize()

    def initialize(self):
        for i in range(2):
            for _ in range(self.n0):
                x = self.blackbox.sample(i)
                self.sample_total[i] += x
                self.sample_sqtot[i] += x ** 2
            self.sample_count[i] = self.n0
            self.mu_hat_curr[i] = self.sample_total[i] / self.sample_count[i]
            # self.pi += -(self.sample_sqtot[i] - self.sample_total[i] * self.mu_hat_curr[i]) / 2

    def sample(self):
        for i in range(2):
            x = self.blackbox.sample(i)
            self.sample_total[i] += x
            self.sample_sqtot[i] += x ** 2
            self.sample_count[i] += 1
            self.mu_hat_prev[i] = self.mu_hat_curr[i]
            self.mu_hat_curr[i] = self.sample_total[i] / self.sample_count[i]

    def update_pi(self):
        self.pi = 0
        for i in range(2):
            self.pi += -(self.sample_sqtot[i] - 2 * self.sample_total[i] * self.mu_hat_prev[i]
                         + self.sample_count[i] * self.mu_hat_prev[i] ** 2) / 2

    def update_ll(self, i):

        mu0, mu1 = self.mu_hat_curr

        if i == 0:
            if mu0 - mu1 <= self.delta:
                mu0, mu1 = (mu0 + mu1 + delta) / 2, (mu0 + mu1 - delta) / 2
        if i == 1:
            if mu1 - mu0 <= self.delta:
                mu1, mu0 = (mu0 + mu1 + delta) / 2, (mu0 + mu1 - delta) / 2

        ll = 0
        for i in range(2):
            if i == 0:
                mu_tmp = mu0
            else:
                mu_tmp = mu1

            ll += -(self.sample_sqtot[i] - 2 * self.sample_total[i] * mu_tmp + self.sample_count[i] * mu_tmp ** 2) / 2

        return ll

    def iterate(self):
        self.sample()
        self.update_pi()

        for i in range(2):
            self.ll[i] = self.update_ll(i)
            self.glr[i] = self.ll[i] - self.pi - np.log(self.alpha)

        return None

    def run(self):

        cnt = 0

        while True:
            self.iterate()
            cnt += 1

            if any([v < 0 for v in self.glr]):
                break

        if self.glr[0] < 0 and self.glr[1] < 0:
            flag = 'Reject Both'
        elif self.glr[0] < 0:
            flag = 'Reject 0'
        else:
            flag = 'Reject 1'

        return {
            'sample_size': np.sum(self.sample_count),
            'flag': flag,
        }


if __name__ == '__main__':

    num_iter = 1000
    delta = 0

    lst_alpha = [0.95, 0.975, 0.9875]

    for alpha in lst_alpha:
        for diff in [8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625]:
            output = []
            # for _ in tqdm.tqdm(range(100)):
            for _ in tqdm.tqdm(range(num_iter)):
                solver = Procedure2Sys(Blackbox2Sys(diff=diff), alpha=1-alpha, delta=delta, n0=2)
                output.append(solver.run())
                # print(output[-1])
            print("")
            print("*" * 10)
            print("alpha: {}, delta: {}, diff: {}, runs: {}".format(alpha, delta, diff, num_iter))
            print("avg_sample_size: {}".format(np.mean([op['sample_size'] for op in output])))
            print("std_sample_size: {}".format(np.std([op['sample_size'] for op in output]) / np.sqrt(num_iter) * 1.98))
            print("output: {}".format(Counter([op['flag'] for op in output])))
