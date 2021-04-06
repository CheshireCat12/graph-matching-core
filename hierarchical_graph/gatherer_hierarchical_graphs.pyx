
cdef class GathererHierarchicalGraphs:

    def __init__(self, dict coordinator_params, list percentages, str centrality_measure):
        # Retrieve graphs with labels
        self.coordinator = CoordinatorClassifier(**coordinator_params)

        self.graphs_train, self.labels_train = self.coordinator.train_split(conv_lbl_to_code=True)
        self.graphs_val, self.labels_val = self.coordinator.val_split(conv_lbl_to_code=True)
        self.graphs_test, self.labels_test = self.coordinator.test_split(conv_lbl_to_code=True)

        # Set the graph hierarchical
        measure = MEASURES[centrality_measure]
        self.h_graphs_train = HierarchicalGraphs(self.graphs_train, measure, percentage_hierarchy=percentages)
        self.h_graphs_val = HierarchicalGraphs(self.graphs_val, measure, percentage_hierarchy=percentages)
        self.h_graphs_test = HierarchicalGraphs(self.graphs_test, measure, percentage_hierarchy=percentages)