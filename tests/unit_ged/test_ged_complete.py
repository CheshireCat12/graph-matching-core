import pytest
import pickle

def load_df(filename):
    with open(filename, mode='rb') as file:
        df = pickle.load(file)

    return df

results_letter = './results/goal/anthony_ged_dist_mat_alpha_node_cost0.9_edge_cost2.3.pkl'
results_AIDS = './results/goal/anthony_ged_dist_mat_alpha_node_cost1.1_edge_cost0.1.pkl'
results_mutagenicity = './results/goal/anthony_ged_dist_mat_alpha_node_cost11.0_edge_cost1.1.pkl'

my_results_letter = './results/complete_ged/letter/result_cost_node0.9_cost_edge2.3_alpha1.0.pkl'
my_results_AIDS = './results/complete_ged/AIDS/result_cost_node1.1_cost_edge0.1_alpha1.0.pkl'
my_results_mutagenicity = './results/complete_ged/mutagenicity/result_cost_node11.0_cost_edge1.1_alpha1.0.pkl'


@pytest.mark.skip()
@pytest.mark.parametrize('results, my_results, epsilon',
                         [
                          (results_letter, my_results_letter, 1e-6),
                          (results_AIDS, my_results_AIDS, 1e-9),
                          (results_mutagenicity, my_results_mutagenicity, 1e-9),
                          ])
def test_all(results, my_results, epsilon):
    df_expected = load_df(results)
    df_results = load_df(my_results)

    counter_different_result = 0
    labels = df_results.columns
    for lbl_src in labels:
        for lbl_trgt in labels:
            expected = df_expected[lbl_src][lbl_trgt]
            actual_result = df_results[lbl_src][lbl_trgt]
            if abs(expected - actual_result) > epsilon:
                print(f'{lbl_src} - {lbl_trgt}: {expected}; {actual_result}; diff {expected-actual_result}')
                counter_different_result += 1
                assert False

    assert counter_different_result == 0


# @pytest.mark.skip()
# def test_complete_ged_letter():
#     results_filename = './results/goal/anthony_ged_dist_mat_alpha_node_cost0.9_edge_cost2.3.pkl'
#     my_results_filename = './results/complete_ged/letter/result_cost_node0.9_cost_edge2.3_alpha1.0.pkl'
#
#     df_res = load_df(results_filename)
#     df_my_res = load_df(my_results_filename)
#
#     epsilon = 1e-6
#     counter_different_result = 0
#     labels = df_my_res.columns
#     for label1 in labels:
#         for label2 in labels:
#             res_1_2 = df_res[label1][label2]
#             my_res_1_2 = df_my_res[label1][label2]
#             if abs(res_1_2 - my_res_1_2) > epsilon:
#                 # print(f'{label1} - {label2}: {res_1_2}; {my_res_1_2}, diff{res_1_2-my_res_1_2}')
#                 counter_different_result += 1
#
#
#     assert counter_different_result == 0
#
#
# # @pytest.mark.skip()
# def test_complete_ged_AIDS():
#     results_filename = './results/goal/anthony_ged_dist_mat_alpha_node_cost1.1_edge_cost0.1.pkl'
#     my_results_filename = './results/complete_ged/mutagenicity/result_cost_node11.0_cost_edge1.1_alpha1.0.pkl'
#
#     df_res = load_df(results_filename)
#     df_my_res = load_df(my_results_filename)
#
#     # print(df_res)
#     # print(df_my_res)
#
#     counter_different_result = 0
#     epsilon = 1e-9
#     labels = df_my_res.columns
#     for label1 in labels:
#         for label2 in labels:
#             res_1_2 = df_res[label1][label2]
#             my_res_1_2 = df_my_res[label1][label2]
#
#             if abs(res_1_2 - my_res_1_2) > epsilon:
#                 print(f'    {label1} - {label2}: Mathias: {res_1_2}; Anthony {my_res_1_2}, diff {res_1_2-my_res_1_2}')
#                 print(f'    other {df_res[label2][label1]} - {df_my_res[label2][label1]}')
#                 counter_different_result += 1
#
#     print(counter_different_result)
#
#     assert counter_different_result == 0
#
#     # assert False
#
# # @pytest.mark.skip()
# def test_complete_ged_mutagenicity():
#     results_filename = './results/goal/anthony_ged_dist_mat_alpha_node_cost11.0_edge_cost1.1.pkl'
#     my_results_filename = './results/goal/res_mutagenicity_cost_node11.0_cost_edge1.1.pkl'
#
#     df_res = load_df(results_filename)
#     df_my_res = load_df(my_results_filename)
#     counter_different_result = 0
#     epsilon = 1e-9
#     labels = df_my_res.columns
#     for label1 in labels:
#         for label2 in labels:
#             res_1_2 = df_res[label1][label2]
#             my_res_1_2 = df_my_res[label1][label2]
#
#             if abs(res_1_2 - my_res_1_2) > epsilon:
#                 # print(f'{label1} - {label2}: {res_1_2}; {my_res_1_2}, diff{res_1_2 - my_res_1_2}')
#                 # print(f'other {df_res[label2][label1]} - {df_my_res[label2][label1]}')
#                 counter_different_result += 1
#
#     assert counter_different_result == 0<
