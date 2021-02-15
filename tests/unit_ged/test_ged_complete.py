import pytest
import pickle
from graph_pkg.coordinator import Coordinator

def test_complete_ged_letter():
    results_filename = 'anthony_ged_dist_mat_alpha_node_cost0.9_edge_cost2.3.pkl'
    with open(results_filename, mode='rb') as file:
        df = pickle.load(file)

    coordinator = Coordinator('letter_high', (0.9, 0.9, 2.3, 2.3, 'euclidean'), '.')

    