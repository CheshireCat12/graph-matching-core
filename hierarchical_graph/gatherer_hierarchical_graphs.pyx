import random
from random import shuffle, seed


cdef class GathererHierarchicalGraphs:

    def __init__(self, dict coordinator_params, list percentages,
                 str centrality_measure, bint activate_aggregation=True,
                 bint verbose=False, bint full_dataset=True):
        self.percentages = percentages
        # Retrieve graphs with labels
        self.coordinator = CoordinatorClassifier(**coordinator_params)

        self.graphs_train, self.labels_train = self.coordinator.train_split(conv_lbl_to_code=True)
        self.graphs_val, self.labels_val = self.coordinator.val_split(conv_lbl_to_code=True)
        self.graphs_test, self.labels_test = self.coordinator.test_split(conv_lbl_to_code=True)

        self.aggregation_graphs = self.graphs_train + self.graphs_val
        self.aggregation_labels = self.labels_train + self.labels_val

        if not full_dataset:
            print('Work with a subset')
            seed(42)
            random.shuffle(self.graphs_train)
            random.shuffle(self.graphs_val)
            random.shuffle(self.graphs_test)
            self.graphs_train = self.graphs_train[:200]
            self.graphs_val = self.graphs_val[:80]
            self.graphs_test = self.graphs_test[:80]

            seed(42)
            random.shuffle(self.labels_train)
            random.shuffle(self.labels_val)
            random.shuffle(self.labels_test)
            self.labels_train = self.labels_train[:200]
            self.labels_val = self.labels_val[:80]
            self.labels_test = self.labels_test[:80]

        # Set the graph hierarchical
        self.measure = MEASURES[centrality_measure]
        self.h_graphs_train = HierarchicalGraphs(self.graphs_train, self.measure,
                                                 percentage_hierarchy=percentages,
                                                 verbose=verbose)
        self.h_graphs_val = HierarchicalGraphs(self.graphs_val, self.measure,
                                               percentage_hierarchy=percentages,
                                               verbose=verbose)
        self.h_graphs_test = HierarchicalGraphs(self.graphs_test, self.measure,
                                                percentage_hierarchy=percentages,
                                                verbose=verbose)

        if activate_aggregation:
            self.h_aggregation_graphs = HierarchicalGraphs(self.aggregation_graphs, self.measure,
                                                           percentage_hierarchy=percentages,
                                                           verbose=verbose)


    cpdef list k_fold_validation(self, int cv=5):
        cdef:
            list graphs, labels, folds
        graphs = self.graphs_train + self.graphs_val
        labels = self.labels_train + self.labels_val

        # graphs = graphs[:500]
        # labels = labels[:500]

        seed(42)
        shuffle(graphs)
        seed(42)
        shuffle(labels)

        size_fold = len(graphs) // cv

        folds = []

        for idx in range(cv):
            lower_idx, upper_idx = idx*size_fold, (idx+1)*size_fold
            graphs_train_fold = graphs[:lower_idx] + graphs[upper_idx:]
            labels_train_fold = labels[:lower_idx] + labels[upper_idx:]

            graphs_val1_fold = graphs[lower_idx:upper_idx]
            labels_val_fold = labels[lower_idx:upper_idx]

            h_graphs_train_fold = HierarchicalGraphs(graphs_train_fold, self.measure, self.percentages)
            h_graphs_val1_fold = HierarchicalGraphs(graphs_val1_fold, self.measure, self.percentages)

            folds.append((h_graphs_train_fold, labels_train_fold, h_graphs_val1_fold, labels_val_fold))

        return folds
