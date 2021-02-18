import pickle

def main():
    # filename = './data/goal/anthony_ged_dist_mat_alpha_node_cost1.1_edge_cost0.1.pkl'
    filename = './results/complete_ged/AIDS/res_letter_cost_node0.9_cost_edge2.3.pkl'
    with open(filename, mode='rb') as file:
        df = pickle.load(file)

    print(df)

if __name__ == '__main__':
    main()

