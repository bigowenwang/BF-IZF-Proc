import numpy as np
import tqdm
import pickle

"""
solver for procedure 2, known variance
"""


class Procedure:
    """
    know variance normal distribution
    """

    def __init__(self, blackbox, alpha=0.05, n0=10):

        self.blackbox = blackbox

        self.alpha = alpha
        self.delta = delta

        self.lst_std = blackbox.sd_oracle

        self.n0 = n0

        self.set_orig = set([i for i in range(self.blackbox.n_sys)])
        self.set_candidate = set([i for i in range(self.blackbox.n_sys)])

        self.sample_total = [0] * self.blackbox.n_sys
        self.sample_sqtot = [0] * self.blackbox.n_sys
        self.sample_count = [0] * self.blackbox.n_sys

        self.sample_total_n0 = [0] * self.blackbox.n_sys
        self.sample_sqtot_n0 = [0] * self.blackbox.n_sys
        self.sample_count_n0 = [0] * self.blackbox.n_sys

        self.mu_hat_prev = [0 for _ in range(self.blackbox.n_sys)]
        self.mu_hat_curr = [0 for _ in range(self.blackbox.n_sys)]

        self.pi = 0
        self.ll_common = 0
        self.ll_unique = [0 for _ in range(self.blackbox.n_sys)]

        self.initialize()

    def initialize(self):
        # for i in self.set_candidate:
        for i in range(self.blackbox.n_sys):
            for _ in range(self.n0):
                x = self.blackbox.sample(i)
                self.sample_total[i] += x
                self.sample_sqtot[i] += x ** 2
            self.sample_count[i] = self.n0
            self.mu_hat_curr[i] = self.sample_total[i] / self.sample_count[i]

    def sample(self):
        for i in self.set_candidate:
            x = self.blackbox.sample(i)
            self.sample_total[i] += x
            self.sample_sqtot[i] += x ** 2
            self.sample_count[i] += 1
            self.mu_hat_prev[i] = self.mu_hat_curr[i]
            self.mu_hat_curr[i] = self.sample_total[i] / self.sample_count[i]

            self.sample_total_n0[i] += x
            self.sample_sqtot_n0[i] += x ** 2
            self.sample_count_n0[i] += 1

    def update_pi(self):
        for i in self.set_candidate:
            self.pi += -(self.sample_total[i] - self.sample_count[i] * self.mu_hat_prev[i]) ** 2 / self.lst_std[i] ** 2 / 2

        self.pi = 0
        for i in self.set_candidate:
            self.pi += -(self.sample_sqtot_n0[i] - 2 * self.sample_total_n0[i] * self.mu_hat_prev[i] +
                                self.sample_count_n0[i] * self.mu_hat_prev[i] ** 2) / self.lst_std[i] ** 2 / 2

        self.ll_common = 0
        for i in self.set_candidate:
            self.ll_common += -(self.sample_sqtot_n0[i] - 2 * self.sample_total_n0[i] * self.mu_hat_curr[i] +
                                self.sample_count_n0[i] * self.mu_hat_curr[i] ** 2) / self.lst_std[i] ** 2 / 2

        # print(self.ll_common - self.pi)

    def update_ll(self, i):

        ll = self.ll_common

        num = 0
        den = 0

        for j in self.set_candidate:
            if self.mu_hat_curr[j] >= self.mu_hat_curr[i] - self.delta:
                num += self.sample_total[j]
                den += self.sample_count[j]

        mu_bar = num / den - delta

        for j in self.set_candidate:
            if self.mu_hat_curr[j] >= self.mu_hat_curr[i]:
                if i == j:
                    mu_tmp = mu_bar + self.delta
                else:
                    mu_tmp = mu_bar
                ll -= -(self.sample_sqtot_n0[i] - 2 * self.sample_total_n0[i] * self.mu_hat_curr[i] +
                        self.sample_count_n0[i] * self.mu_hat_curr[i] ** 2) / self.lst_std[i] ** 2 / 2
                ll += -(self.sample_sqtot_n0[i] - 2 * self.sample_total_n0[i] * mu_tmp +
                        self.sample_count_n0[i] * mu_tmp ** 2) / self.lst_std[i] ** 2 / 2

        self.ll_unique[i] = ll

    def iterate(self):
        self.sample()
        self.update_pi()

        glr = [-1 for _ in range(self.blackbox.n_sys)]

        imax = self.mu_hat_curr.index(max(self.mu_hat_curr))

        for i in self.set_candidate:
            self.update_ll(i)
            glr[i] = self.ll_unique[i] - self.pi - np.log(self.alpha)

        self.set_candidate -= set([i for i in self.set_candidate if glr[i] < 0 and i != imax])

    def run(self):

        cnt = 0

        while len(self.set_candidate) > 1:
            self.iterate()
            cnt += 1

        # print(self.set_candidate)
        # print(self.mu_hat_curr)

        return {
            'sample_size': np.sum(self.sample_count),
            'is_best': True if 0 in self.set_candidate else False,
            'is_good': any(ele <= self.delta for ele in self.set_candidate),
            'candid': self.set_candidate,
        }


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
        return np.random.normal(self.mu_oracle[i], self.sd_oracle[i])


if __name__ == '__main__':

    num_iter = 1000
    delta = 0.5

    # for delta in lst_delta:
    for delta in [0, 0.5]:
        for n_sys in [20, 50, 100]:
            for v in ['e', 'i', 'd']:
                bb = Blackbox3(n_sys=n_sys, setting=v)
                output = []
                for _ in tqdm.tqdm(range(num_iter)):
                    solver = Procedure(bb, alpha=0.02, n0=10)
                    # print(solver.mu_hat_curr)
                    output.append(solver.run())

                print("")
                print("*" * 10)
                print("n_sys: {}, delta: {}, setting: {}".format(n_sys, delta, v))
                print("avg_sample_size: {}".format(np.mean([op['sample_size'] for op in output])))
                print("std_sample_size: {}".format(np.std([op['sample_size'] for op in output]) / np.sqrt(num_iter) * 1.98))
                print("avg_pcs: {}".format(np.mean([1 * op['is_best'] for op in output])))
                print("avg_pgs: {}".format(np.mean([1 * op['is_good'] for op in output])))

                file = open(f"{n_sys}_{delta}_{v}.pkl", "wb")
                pickle.dump(output, file)
                file.close()