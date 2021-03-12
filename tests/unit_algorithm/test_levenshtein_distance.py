import numpy as np
from graph_pkg.algorithm.levenshtein_distance import LevenshteinDistance

levi = LevenshteinDistance()

def test_levenshtein_equal_length_strings():
    string_1 = "ABABBB"
    string_2 = "BABAAA"
    # example from PR lecture
    expected_edit_distance_matr = [[0, 1, 2, 3, 4, 5, 6],
                                   [1, 2, 1, 2, 3, 4, 5],
                                   [2, 1, 2, 1, 2, 3, 4],
                                   [3, 2, 1, 2, 1, 2, 3],
                                   [4, 3, 2, 1, 2, 3, 4],
                                   [5, 4, 3, 2, 3, 4, 5],
                                   [6, 5, 4, 3, 4, 5, 6]]
    expected_edit_distance = 6

    actual_distance = levi.compute_string_edit_distance_cpd(string_1, string_2, 2, 1, 1)
    actual_distance_matr = np.asarray(levi.distances)

    assert np.allclose(expected_edit_distance_matr, actual_distance_matr)
    assert expected_edit_distance == actual_distance


def test_levensthein_not_equal_length_strings():
    string_1 = "BAB"
    string_2 = "ABAO"
    expected_edit_distance_matr = [[0, 1, 2, 3, 4],
                                   [1, 1, 1, 2, 3],
                                   [2, 1, 2, 1, 2],
                                   [3, 2, 1, 2, 2]]
    expected_edit_distance = 2

    actual_distance = levi.compute_string_edit_distance_cpd(string_1, string_2, 1, 1, 1)
    actual_distance_matr = np.asarray(levi.distances)

    assert np.allclose(expected_edit_distance_matr, actual_distance_matr)
    assert expected_edit_distance == actual_distance


def test_levensthein_empty_strings():
    string_1 = "ABAO"
    string_2 = ""
    expected_edit_distance_matr = [[0],
                                   [1],
                                   [2],
                                   [3],
                                   [4]]
    expected_edit_distance = 4

    actual_distance = levi.compute_string_edit_distance_cpd(string_1, string_2, 1, 1, 1)
    actual_distance_matr = np.asarray(levi.distances)

    assert np.allclose(expected_edit_distance_matr, actual_distance_matr)
    assert expected_edit_distance == actual_distance


def test_unequal_costs():
    string_1 = "BXB"
    string_2 = "ABAO"
    expected_edit_distance_matr = [[0, 1, 2, 3, 4],
                                   [1, 2, 1, 3, 5],
                                   [2, 3, 2, 4, 6],
                                   [3, 4, 3, 5, 7]]
    expected_edit_distance = 7

    actual_distance = levi.compute_string_edit_distance_cpd(string_1, string_2, 3,2,1)
    actual_distance_matr = np.asarray(levi.distances)

    assert np.allclose(expected_edit_distance_matr, actual_distance_matr)
    assert expected_edit_distance == actual_distance
