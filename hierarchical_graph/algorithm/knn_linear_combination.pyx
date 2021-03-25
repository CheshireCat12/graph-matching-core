import numpy as np
cimport numpy as np
from collections import Counter
from itertools import product

from graph_pkg.utils.constants cimport PERCENT_HIERARCHY


cdef class KNNLinearCombination:
    """
    Basic KNN classifier.
    Look for the k closest neighbors of a data point.
    Returns the majority class of the k neighbors.
    """

    def __init__(self, GED ged, bint parallel=False):
        """
        Initialize the KNN with the graph edit distance class.
        The ged is used to compute the distance between 2 graphs.

        :param ged: GED
        """
        self.ged = ged
        self.mat_dist = MatrixDistances(ged, parallel)

    cpdef void train(self, HierarchicalGraphs h_graphs_train, list labels_train):
        """
        Function used to train the KNN classifier.
        With KNN, the training step consists to save the training set.
        The saved training set is used later during the prediction step.

        :param X_train: list
        :param y_train: list
        :return: 
        """
        self.h_graphs_train = h_graphs_train
        self.labels_train = labels_train
        self.np_labels_train = np.array(labels_train, dtype=np.int32)

    cpdef double[:, :, ::1] _get_distances(self, HierarchicalGraphs h_graphs_pred, int size_pred_set):

        shape = (len(PERCENT_HIERARCHY), len(self.labels_train), size_pred_set)
        h_distances = np.empty(shape)

        for idx, percentage in enumerate(PERCENT_HIERARCHY):
            h_distances[idx] = self.mat_dist.calc_matrix_distances(self.h_graphs_train.hierarchy[percentage],
                                                                   h_graphs_pred.hierarchy[percentage],
                                                                   heuristic=True)

        return h_distances

    cpdef tuple optimize(self, HierarchicalGraphs h_graphs_pred,
                         list labels_pred,
                         int k,
                         str optimization_strategy='linear'):
        """
        Predict the class for the graphs in X.
        It returns the majority of the k nearest neighbor from the trainset.

        - First it computes the distance matrix between the training set and the test set.
        - Second it finds the k nearest points for the graphs in the test set.
        - Take the majority class from the k nearest point of the training set.

        :param graphs_pred: list
        :param k: int
        :return: predictions: list
        """
        cdef:
            list alphas
            int[::1] np_labels_pred
            double[:, ::1] distances, tmp_distances
            double[:, :, ::1] h_distances

        print('\n-- Start prediction --')
        np_labels_pred = np.array(labels_pred, dtype=np.int32)

        h_distances = self._get_distances(h_graphs_pred, len(labels_pred))


        if optimization_strategy == 'linear':
            alphas = [i/10 for i in range(1, 10)]
            population = np.array(list(product(alphas, repeat=len(PERCENT_HIERARCHY))))
            accuracies = self.fitness(population, h_distances, np_labels_pred, k)

            idx_best_acc = np.argmax(accuracies)

            return accuracies[idx_best_acc], population[idx_best_acc]

        elif optimization_strategy == 'genetic':
            np.random.seed(6)

            p_crossover = 0.85
            p_mutation = 0.1

            size_population = 50
            population = np.random.rand(size_population, len(PERCENT_HIERARCHY))
            population = np.around(population, 3)

            best_acc = float('-inf')
            best_alphas = None
            gamma = 0.51
            for _ in range(50):
                accuracies = self.fitness(population, h_distances, np_labels_pred, k)

                idx_max_acc = np.argmax(accuracies)
                if accuracies[idx_max_acc] > best_acc:
                    best_acc = accuracies[idx_max_acc]
                    best_alphas = population[idx_max_acc]
                    print('')
                    print(f'## Best of Best {best_acc}, alpha: {best_alphas}')
                    print('')

                prob = accuracies / np.sum(accuracies)

                # tmp = list(range(len(population)))
                idx_parents = np.random.choice(len(population),
                                               len(population),
                                               p=prob)
                # print(idx_parents)
                #
                # print(sorted(idx_parents))
                new_population = []

                for idx in range(0, len(population), 2):
                    idx_parent1, idx_parent2 = idx_parents[idx], idx_parents[idx+1]
                    parent1, parent2 = population[idx_parent1], population[idx_parent2]
                    child1, child2 = np.zeros(parent1.shape), np.zeros(parent2.shape)

                    # Crossover
                    for idx_gene, (gene1, gene2) in enumerate(zip(parent1,parent2)):
                        child1[idx_gene] = gamma * gene1 + (1-gamma) * gene2
                        child2[idx_gene] = gamma * gene1 + (1-gamma) * gene2 # (1-gamma) * gene1 + gamma * gene2
                        # if np.random.randint(2) == 0:
                        #     child1[idx_gene] = gene1
                        #     child2[idx_gene] = gene2
                        # else:
                        #     child1[idx_gene] = gene2
                        #     child2[idx_gene] = gene1

                    if np.random.rand() <= p_mutation:
                        idx_to_mutate = np.random.choice(len(child1))
                        child1[idx_to_mutate] = np.random.rand()
                    if np.random.rand() <= p_mutation:
                        idx_to_mutate = np.random.choice(len(child2))
                        child2[idx_to_mutate] = np.random.rand()

                    new_population.append(child1)
                    new_population.append(child2)

                #     break
                #
                # break
                population = np.around(np.array(new_population), 3)

            return best_acc, best_alphas
            # idx_best_acc = np.argmax(accuracies)
            #
            # return accuracies[idx_best_acc], population[idx_best_acc]



    cpdef double[::1] fitness(self, double[:, ::1] population,
                              double[:, :, ::1] h_distances,
                              int[::1] np_labels_test,
                              int k):
        cdef:
            double[::1] accuracies = np.zeros((population.shape[0],))

        best_acc = float('-inf')

        for idx_gene, gene in enumerate(population):

            # Create distances matrix
            dim1, dim2 = h_distances.shape[1:3]
            distances = np.zeros((dim1, dim2))
            for idx, _ in enumerate(PERCENT_HIERARCHY):
                distances += np.array(h_distances[idx, :, :]) * gene[idx]

            # Get the index of the k smallest distances in the matrix distances.
            idx_k_nearest = np.argpartition(distances, k, axis=0)[:k]

            # Get the label of the k smallest distances.
            labels_k_nearest = np.asarray(self.np_labels_train)[idx_k_nearest]

            # Get the array of the prediction of all the elements of X.
            prediction = np.array([Counter(arr).most_common()[0][0]
                                   for arr in labels_k_nearest.T])

            correctly_classified = np.sum(prediction == np_labels_test)
            accuracy = 100 * (correctly_classified / len(np_labels_test))

            if accuracy > best_acc:
                print(f'Accuracy {accuracy:.2f}, alpha: {np.asarray(gene)}')
                best_acc = accuracy

            accuracies[idx_gene] = accuracy

        return accuracies

    cpdef double predict(self, HierarchicalGraphs h_graphs_pred,
                         list labels_pred,
                         int k,
                         double[::1] alphas):
        cdef:
            # list alphas
            int[::1] np_labels_pred
            double[:, ::1] distances, tmp_distances
            double[:, :, ::1] h_distances

        print('\n-- Start prediction --')
        np_labels_pred = np.array(labels_pred, dtype=np.int32)

        h_distances = self._get_distances(h_graphs_pred, len(labels_pred))

        accuracies = self.fitness(np.array([alphas,]), h_distances, np_labels_pred, k)

        return accuracies[0]

