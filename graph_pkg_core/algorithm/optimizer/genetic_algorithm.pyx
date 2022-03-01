import numpy as np
cimport numpy as np

cdef class GeneticAlgorithm(Optimizer):

    def __init__(self, double range_down, range_up, int size, int n_genes, int optimization_turn=1):
        super().__init__(range_down, range_up, size, optimization_turn)

        np.random.seed(6)
        self.p_crossover = 0.85
        self.p_mutation = 0.1
        self.round_val = 4
        self.gamma = 0.5

        self.opt_values = np.random.uniform(range_down, range_up, (n_genes, size))
        self.opt_values = np.around(self.opt_values, self.round_val)
        self.accuracies = np.zeros(len(self.opt_values))

    cpdef void update_values(self):
        cdef:
            list new_population
            double[::1] prob

        print(f'max acc per turn {max(np.array(self.accuracies))}')

        # prob = self.accuracies / np.sum(self.accuracies)

        test_prob = self.accuracies - (min(np.array(self.accuracies)) - 0.00001)
        prob = test_prob / sum(test_prob)

        idx_parents = np.random.choice(len(self.opt_values),
                                       len(self.opt_values),
                                       p=prob)

        new_population = []
        distribution_index = 3

        for idx in range(0, len(self.opt_values), 2):
            idx_parent1, idx_parent2 = idx_parents[idx], idx_parents[idx+1]
            parent1, parent2 = self.opt_values[idx_parent1], self.opt_values[idx_parent2]
            child1, child2 = np.zeros(parent1.shape), np.zeros(parent2.shape)

            # Crossover
            for idx_gene, (gene1, gene2) in enumerate(zip(parent1,parent2)):
                # u = np.random.rand()
                # if u <= 0.5:
                #     beta = np.power(2*u, 1/(distribution_index + 1))
                # else:
                #     beta = np.power(1/(2*(1-u)), 1/(distribution_index + 1))
                # new_gene1 = 0.5 * ((1 + beta) * gene1 + (1 - beta) * gene2)
                # new_gene2 = 0.5 * ((1 - beta) * gene1 + (1 + beta) * gene2)
                # child1[idx_gene] = min(max(new_gene1, 0.), 1.0)
                # child2[idx_gene] = min(max(new_gene2, 0.), 1.0)


                low_range = max(min(gene1, gene2) - self.gamma * abs(gene1 - gene2), 0.0)
                high_range = min(max(gene1, gene2) + self.gamma * abs(gene1 - gene2), 1.0)
                # child1[idx_gene] = np.random.uniform(low=low_range, high=high_range)
                # child2[idx_gene] = np.random.uniform(low=low_range, high=high_range)


                child1[idx_gene] = self.gamma * gene1 + (1-self.gamma) * gene2
                child2[idx_gene] = self.gamma * gene1 + (1-self.gamma) * gene2 # (1-gamma) * gene1 + gamma * gene2

                # if np.random.randint(2) == 0:
                #     child1[idx_gene] = gene1
                #     child2[idx_gene] = gene2
                # else:
                #     child1[idx_gene] = gene2
                #     child2[idx_gene] = gene1

                #mutation
                if np.random.rand() <= self.p_mutation:
                    child1[idx_gene] = np.random.rand()
                    # child1[idx_gene] = np.random.uniform(low=low_range, high=high_range)
                if np.random.rand() <= self.p_mutation:
                    child2[idx_gene] = np.random.rand()
                    # child2[idx_gene] = np.random.uniform(low=low_range, high=high_range)

            new_population.append(child1)
            new_population.append(child2)

        self.opt_values = np.around(np.array(new_population), self.round_val)


