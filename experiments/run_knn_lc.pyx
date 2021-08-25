from progress.bar import Bar
import pygad
import numpy as np
cimport numpy as np

from hierarchical_graph.algorithm.knn_linear_combination cimport KNNLinearCombination as KNNLC
from experiments.runner import Runner
from hierarchical_graph.gatherer_hierarchical_graphs cimport GathererHierarchicalGraphs as GAG
from graph_pkg.utils.functions.helper import calc_accuracy, calc_f1
from graph_pkg.algorithm.optimizer.optimizer cimport Optimizer
from graph_pkg.algorithm.optimizer.grid_search cimport GridSearch
from graph_pkg.algorithm.optimizer.genetic_algorithm cimport GeneticAlgorithm



class RunnerKnnLC(Runner):

    def __init__(self, parameters):
        super(RunnerKnnLC, self).__init__(parameters)

    def run(self):
        print('Run KNN with Linear Combination')

        if self.parameters.optimize:
            best_params = self.optimization()

            acc_on_test = self.evaluate(best_params)
        else:
            best_params = self.parameters.knn_lc_params.values()

            self.evaluate(best_params)

    def optimization(self):
        """
        Optimize the hyperparmaters of the linear combination.
        Try to find the best combinations of distances to improve the overall accuracy of the model.

        :return:
        """
        cdef:
            Optimizer optimizer
            KNNLC knn_lc

        ##################
        # Set parameters #
        ##################
        params_edit_cost = self.parameters.coordinator['params_edit_cost']
        best_alpha = self.parameters.best_alpha
        self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, best_alpha)

        coordinator_params = self.parameters.coordinator
        dataset = self.parameters.coordinator['dataset']
        centrality_measure = self.parameters.centrality_measure
        deletion_strategy = self.parameters.deletion_strategy
        k = self.parameters.k
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel
        percentages = self.parameters.hierarchy_params['percentages']
        cv = self.parameters.cv

        gag = GAG(coordinator_params, percentages, centrality_measure)

        knn_lc = KNNLC(gag.coordinator.ged, k, parallel)

        if self.parameters.optimization_strategy == 'grid_search':
            optimizer = GridSearch(0.0, 1.0, 5, 1)

            return self.grid_search(knn_lc, gag, optimizer, cv, num_cores, dataset)
        elif self.parameters.optimization_strategy == 'genetic_algorithm':

            optimizer = GeneticAlgorithm(0.0, 1.0, 5, n_genes=50, optimization_turn=50)

            return self.ga(knn_lc, gag, num_cores, dataset)
        else:
            raise NotImplementedError('Optimizer not implemented')

    def ga(self, knn_lc, gag, num_cores, dataset):

        # num_generations = 100
        # num_parents_mating = 2
        # sol_per_pop = 50
        # num_genes = 5

        # Train the classifier
        knn_lc.train(gag.h_graphs_train, gag.labels_train)

        # Compute the distances in advance not to have to compute it every turn
        knn_lc.load_h_distances(gag.h_graphs_val, folder_distances='',
                                is_test_set=False,
                                num_cores=num_cores)

        labels_val = gag.labels_val

        def fitness_func(solution, solution_idx):
            predictions = knn_lc.predict_dist(solution)
            # else:
            # predictions = knn_lc.compute_pred_from_score(overall_predictions, omegas)

            acc = calc_accuracy(np.array(gag.labels_val, dtype=np.int32), predictions)
            # print(acc)
            return acc

        def callback_generation(ga_instance):
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

        ga_params = self.parameters.ga_params
        ga_params['num_genes'] = len(gag.h_graphs_train.hierarchy.keys())
        ga_params['fitness_func'] = fitness_func
        ga_instance = pygad.GA(**ga_params)

        ga_instance.run()

        ga_instance.plot_fitness()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

        ga_params['fitness_func'] = 'func'

        return [solution]


    def grid_search(self, knn_lc, gag, optimizer, cv, num_cores, dataset):
        accuracies = np.zeros((len(optimizer.opt_values), cv))
        f1_scores = np.zeros((len(optimizer.opt_values), cv))

        for idx, (h_graphs_train, labels_train, h_graphs_val, labels_val) in enumerate(gag.k_fold_validation(cv=cv)):

            # Train the classifier
            knn_lc.train(h_graphs_train, labels_train)

            # Compute the distances in advance not to have to compute it every turn
            knn_lc.load_h_distances(h_graphs_val, folder_distances='',
                                    is_test_set=False,
                                    num_cores=num_cores)

            best_acc = float('-inf')
            best_coeff = None

            if not self.parameters.dist:
                overall_predictions = knn_lc.predict_score()

            bar = Bar(f'Processing, Turn {idx+1}/{cv}', max=len(optimizer.opt_values))
            for idx_coef, omegas in enumerate(optimizer.opt_values):
                omegas = np.array(omegas)

                if self.parameters.dist:
                    predictions = knn_lc.predict_dist(omegas)
                else:
                    predictions = knn_lc.compute_pred_from_score(overall_predictions, omegas)

                acc = calc_accuracy(np.array(labels_val, dtype=np.int32), predictions)
                f1_score = calc_f1(np.array(labels_val, dtype=np.int32), predictions, dataset)

                accuracies[idx_coef][idx] = acc
                optimizer.accuracies[idx_coef] = acc
                f1_scores[idx_coef][idx] = f1_score

                if acc > best_acc:
                    best_acc = acc
                    best_coeff = omegas
                    print('|||', best_acc, omegas)

                bar.next()


            optimizer.update_values()

            print(f'best acc : {best_acc}, best coeff: {best_coeff}')

            bar.finish()


        mean_acc = np.mean(accuracies, axis=1)
        median_acc = np.median(accuracies, axis=1)
        mean_f1_score = np.mean(f1_scores, axis=1)

        idx_best_omega = np.argmax(mean_acc)
        best_omega = np.array(optimizer.opt_values[idx_best_omega])
        max_mean = mean_acc[idx_best_omega]
        # print(max_mean)
        # print(max(mean_acc))
        best_coefficients = []

        message = ''

        for idx, (mean, median, f1) in enumerate(zip(mean_acc, median_acc, mean_f1_score)):
            if mean == max_mean:
                message += f'Mean/Median/F1 Score: {mean}, {median}, {f1}\n' \
                           f'{", ".join(f"{val:.2f}" for val in optimizer.opt_values[idx])}\n' \
                           f'########\n'
                best_coefficients.append(optimizer.opt_values[idx])

        # print(message)
        self.save_stats(message, 'coefficients.txt', save_params=False)


        print(max(mean_acc))
        print(best_omega)

        best_coefficients = [best_coeff]

        return best_coefficients

    def evaluate(self, best_params):
        cdef:
            KNNLC knn_lc

        ##################
        # Set parameters #
        ##################
        if not self.parameters.optimize:
            params_edit_cost = self.parameters.coordinator['params_edit_cost']
            best_alpha = self.parameters.best_alpha
            self.parameters.coordinator['params_edit_cost'] = (*params_edit_cost, best_alpha)

        coordinator_params = self.parameters.coordinator
        centrality_measure = self.parameters.centrality_measure
        deletion_strategy = self.parameters.deletion_strategy
        k = self.parameters.k
        num_cores = self.parameters.num_cores
        parallel = self.parameters.parallel
        percentages = self.parameters.hierarchy_params['percentages']
        cv = self.parameters.cv

        gag = GAG(coordinator_params, percentages, centrality_measure)

        knn_lc = KNNLC(gag.coordinator.ged, k, parallel)



        best_omegas, *_ = best_params
        best_omegas = np.array(best_omegas)
        print('best_omegas')
        print(best_omegas)


        # knn.train(gag.h_aggregation_graphs, gag.aggregation_labels)
        knn_lc.train(gag.h_graphs_train, gag.labels_train)
        knn_lc.load_h_distances(gag.h_graphs_test, self.parameters.folder_results,
                             is_test_set=True, num_cores=num_cores)


        if self.parameters.dist:
            pass
            predictions_final = knn_lc.predict_dist(best_omegas)
            accuracy_final = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
                                       predictions_final)
            message = f'Acc: {accuracy_final:.3f}\n' \
                      f'Linear combination: {best_omegas}'
        else:
            best_acc = float('-inf')
            message = ''
            for best_omegas in best_params:
                overall_predictions = knn_lc.predict_score()
                predictions_final = knn_lc.compute_pred_from_score(overall_predictions, np.array(best_omegas))

                accuracy_final = calc_accuracy(np.array(gag.labels_test, dtype=np.int32),
                                               predictions_final)

                if accuracy_final > best_acc:
                    best_acc = accuracy_final

                message += f'Acc: {accuracy_final:.2f}, Best so far: {best_acc:.2f}\n' \
                           f'Linear combination: {best_omegas}\n'



        filename = f'acc_{"dist" if self.parameters.dist else "score"}'
        print(message)
        self.save_stats(message, name=f'{filename}.txt')

        return accuracy_final


cpdef void run_knn_lc(parameters):
    run_knn_lc = RunnerKnnLC(parameters)
    run_knn_lc.run()
