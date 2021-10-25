import os
from collections import defaultdict
from itertools import product
from pathlib import Path
from time import time

import numpy as np
cimport numpy as np
import pandas as pd
from progress.bar import Bar

from experiments.runner import Runner
from graph_pkg.algorithm.knn cimport KNNClassifier
from graph_pkg.utils.functions.helper import calc_accuracy
from graph_pkg.utils.coordinator_gnn_embedding.coordinator_gnn_embedding_classifier import CoordinatorGNNEmbeddingClassifier

class RunnerKNNGNN(Runner):

    def __init__(self, parameters):
        super(RunnerKNNGNN, self).__init__(parameters)

    def run(self):
        print('Run KNN with Reduced Graphs')

        # Init the graph gatherer
        coordinator_params = self.parameters.coordinator
        print(coordinator_params)

        run_full_dataset = False if self.parameters.coordinator['dataset'] in ['collab', 'reddit_binary'] else True

        self.save_stats('The code is running\n', 'log.txt', save_params=False)

        self.coordinator = CoordinatorGNNEmbeddingClassifier(**coordinator_params)

        self.save_stats('The graphs are loaded\n', 'log.txt')

        self.test_evaluation = []

        if self.parameters.optimize:
            best_params = self.optimization()
            # best_params = (1, 0.7)

            self.evaluate(best_params)

        else:
            params_from_file = self.parameters.h_knn

            # Retrieve the parameters from the full graph optimization
            current_percentage = params_from_file['current_percentage']
            best_params = tuple(params_from_file[current_percentage].values())

            # Select on which percentage test the obtained parameters
            if self.parameters.check_all_percentages:
                percentage_to_check = self.parameters.hierarchy_params['percentages']
            else:
                percentage_to_check = [current_percentage]

            for lambda_ in percentage_to_check:
                self.parameters.current_percentage_to_opt = lambda_

                self.evaluate(best_params)


    def optimization(self):
        cdef:
            KNNClassifier knn

        ##################
        # Set parameters #
        ##################

        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel

        graphs_train, labels_train = self.coordinator.train_split()
        graphs_val, labels_val = self.coordinator.val_split()

        knn = KNNClassifier(self.coordinator.ged, parallel, verbose=False)
        knn.train(graphs_train=graphs_train, labels_train=labels_train)

        # Hyperparameters to tune
        alpha_start, alpha_end, alpha_step = self.parameters.tuning['alpha']
        alphas = [alpha_step * i for i in range(alpha_start, alpha_end)]
        ks = self.parameters.tuning['ks']

        best_acc = float('-inf')
        best_params = (None, None)
        accuracies = defaultdict(list)

        hyperparameters = product(ks, alphas)
        len_hyperparameters = len(ks) * len(alphas)

        bar = Bar(f'Processing Graphs : Optimization', max=len_hyperparameters)

        prediction_times = []

        for k_param, alpha in hyperparameters:
            alpha = round(alpha, 2)
            self.coordinator.edit_cost.update_alpha(alpha)

            start_time = time()
            predictions = knn.predict(graphs_pred=graphs_val, k=k_param, num_cores=num_cores)
            prediction_time = time() - start_time

            prediction_times.append(prediction_time)

            acc = calc_accuracy(np.array(labels_val, dtype=np.int32), predictions)

            if acc >= best_acc:
                best_acc = acc
                best_params = (k_param, alpha)

            accuracies[k_param].append(acc)

            Bar.suffix = f'%(index)d/%(max)d | Best acc {best_acc:.2f}, with {best_params} -' \
                         f' ellapse time: {sum(prediction_times) / len(prediction_times):.2f}s'
            bar.next()
        bar.finish()

        # Save the validation accuracy per hyperparameter
        # Path(self.parameters.folder_results).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(self.parameters.folder_results,
                                f'GNN_embedding_fine_tuning.csv')

        dataframe = pd.DataFrame(accuracies, index=alphas, columns=ks)
        dataframe.to_csv(filename)

        # Save the best acc on validation
        message = f'Best acc on validation : {best_acc:.2f}, best params: {best_params}'
        print(message)
        self.save_stats(message, f'opt_h_knn.txt', save_params=False)

        return best_params


    def evaluate(self, best_params):
        cdef:
            KNNClassifier knn

        print(best_params)

        best_k, best_alpha = best_params
        self.coordinator.edit_cost.update_alpha(best_alpha)

        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel

        graphs_train, labels_train = self.coordinator.train_split()
        # graphs_test, labels_test = self.coordinator.val_split()
        graphs_test, labels_test = self.coordinator.test_split()

        knn = KNNClassifier(self.coordinator.ged, parallel, verbose=False)
        knn.train(graphs_train=graphs_train, labels_train=labels_train)

        start_time = time()
        predictions = knn.predict(graphs_test, k=best_k, num_cores=num_cores)
        prediction_time = time() - start_time

        acc = calc_accuracy(np.array(labels_test, dtype=np.int32), predictions)

        message = f'Best acc on Test : {acc:.2f}, best params: {best_params}, time: {prediction_time}\n'
        print(message)



        if self.parameters.optimize:
            self.save_stats(message, f'test_results_h_knn_.txt')

        # else:
        #     write_params = current_percentage_opt == 1.0
        #     self.save_stats(message, f'{centrality_measure}_test_results_not_opt.txt', save_params=write_params)

        # self.test_evaluation.append((current_percentage_opt, acc, prediction_time))

        # if self.parameters.save_dist_matrix:
        #     # Save the validation accuracy per hyperparameter
        #     folder = os.path.join(self.parameters.folder_results, 'distances')
        #     Path(folder).mkdir(parents=True, exist_ok=True)
        #     filename = os.path.join(folder,
        #                             f'{centrality_measure}_dist_{current_percentage_opt}.npy')
        #
        #     with open(filename, 'wb') as f:
        #         np.save(f, knn.current_distances)

        filename = f'prediction_full'
        self.save_predictions(predictions, np.array(labels_test, dtype=np.int32), f'{filename}.npy')
        # Reinitialize the coordinator params
        # self.parameters.coordinator['params_edit_cost'] = params_edit_cost


cpdef void run_knn_gnn_embedding(parameters):
    run_h_knn_ = RunnerKNNGNN(parameters)
    run_h_knn_.run()
