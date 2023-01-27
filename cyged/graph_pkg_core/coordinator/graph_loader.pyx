import os
import pickle
import re
from glob import glob
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def _load_graphml(graph_files: List[str], node_attr:str ='x') -> List[nx.Graph]:
    """
    Load all the graphs from the list as nx.Graph.
    Use the graphml loader of networkx.

    Args:
        graph_files: A list of graph filenames
        node_attr: The index of the node attribute in the graphs

    returns:
        A list of loaded `nx.Graph`.
    """
    nx_graphs = [None] * len(graph_files)

    for file in tqdm(graph_files, desc='Load Graphs'):

        # The idx of the graph is retrieved from its filename
        filename = file.split('/')[-1]
        idx_graph = int(re.findall('[0-9]+', filename)[0])

        nx_graph = nx.read_graphml(file)

        for idx_node, data_node in nx_graph.nodes(data=True):
            np_data = np.fromstring(data_node[node_attr][1:-1], sep=',')
            nx_graph.nodes[idx_node][node_attr] = np_data

        nx_graphs[idx_graph] = nx_graph

    return nx_graphs

def _load_pkl(graph_files: List[str], *args) -> List[nx.Graph]:

    with open(graph_files[0], 'rb') as f:
        nx_graphs = pickle.load(f)

    return nx_graphs


GRAPH_FORMAT = {'graphml': _load_graphml,
                'pkl': _load_pkl}

def load_graphs(root_dataset: str,
                file_extension: str = 'graphml',
                node_attr: str = 'x',
                load_classes: bool = False,
                file_classes: str = 'graph_classes.csv') -> Tuple[List[nx.Graph], np.ndarray]:
    """
    Args:
        root_dataset:
        file_extension:
        node_attr:
        load_classes:
        file_classes:
    Returns:
    """
    assert file_extension in GRAPH_FORMAT.keys(), 'file extension not valid!'

    graph_files = glob(os.path.join(root_dataset, f'*.{file_extension}'))
    graphs = GRAPH_FORMAT[file_extension](graph_files, node_attr)

    classes = None
    if load_classes:
        classes_file = os.path.join(root_dataset, file_classes)
        df = pd.read_csv(classes_file)
        classes = df['class'].to_numpy()

    return graphs, classes
