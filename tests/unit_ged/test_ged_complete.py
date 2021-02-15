import pytest
import pickle

def load_df(filename):
    with open(filename, mode='rb') as file:
        df = pickle.load(file)

    return df

@pytest.mark.skip()
def test_complete_ged_letter():
    results_filename = './data/goal/anthony_ged_dist_mat_alpha_node_cost0.9_edge_cost2.3.pkl'
    my_results_filename = './data/goal/res_letter_cost_node0.9_cost_edge2.3.pkl'

    df_res = load_df(results_filename)
    df_my_res = load_df(my_results_filename)

    labels = df_my_res.columns
    for label1 in labels[:10]:
        for label2 in labels[:10]:
            print(f'{label1} - {label2}: {df_res[label1][label2]}; {df_my_res[label1][label2]}')
            print(f'{label2} - {label1}: {df_res[label2][label1]}; {df_my_res[label2][label1]}')



    assert False


@pytest.mark.skip()
def test_complete_ged_AIDS():
    results_filename = './data/goal/anthony_ged_dist_mat_alpha_node_cost1.1_edge_cost0.1.pkl'
    my_results_filename = './data/goal/res_AIDS_cost_node1.1_cost_edge0.1.pkl'

    df_res = load_df(results_filename)
    df_my_res = load_df(my_results_filename)

    print(df_res)
    print(df_my_res)

    counter = 0
    labels = df_my_res.columns
    for label1 in labels:
        for label2 in labels:
            val_res12, val_my_res12 = df_res[label1][label2], df_my_res[label1][label2]
            val_res21, val_my_res21 = df_res[label2][label1], df_my_res[label2][label1]
            if val_res12 == val_my_res12 or val_res12 == val_my_res21:
                continue
            if val_res21 == val_my_res12 or val_res21 == val_my_res21:
                continue
            # print(f'{label1} - {label2}: {df_res[label1][label2]}; {df_my_res[label1][label2]}')
            # print(f'{label2} - {label1}: {df_res[label2][label1]}; {df_my_res[label2][label1]}')
            counter += 1

    print(counter)



    assert False


def test_complete_ged_mutagenicity():
    results_filename = './data/goal/anthony_ged_dist_mat_alpha_node_cost11.0_edge_cost1.1.pkl'
    my_results_filename = './data/goal/res_mutagenicity_cost_node11.0_cost_edge1.1.pkl'

    df_res = load_df(results_filename)
    df_my_res = load_df(my_results_filename)

    print(df_res)
    print(df_my_res)
    

    counter = 0
    labels = df_my_res.columns
    for label1 in labels[:10]:
        for label2 in labels[:10]:
            val_res12, val_my_res12 = df_res[label1][label2], df_my_res[label1][label2]
            val_res21, val_my_res21 = df_res[label2][label1], df_my_res[label2][label1]
            if val_res12 == val_my_res12 or val_res12 == val_my_res21:
                continue
            if val_res21 == val_my_res12 or val_res21 == val_my_res21:
                continue
            print(f'{label1} - {label2}: {df_res[label1][label2]}; {df_my_res[label1][label2]}')
            print(f'{label2} - {label1}: {df_res[label2][label1]}; {df_my_res[label2][label1]}')
            counter += 1

    print(counter)



    assert False
