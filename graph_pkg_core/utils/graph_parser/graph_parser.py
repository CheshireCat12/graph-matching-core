"""
Graph parser is a file used to parse the graphs
from https://chrsmrrs.github.io/datasets/docs/datasets/ into a ".gxl" file

@author: Anthony Gillioz
@date: 04.03.2021

"""
import os
from pathlib import Path
from collections import defaultdict, namedtuple
import xml.etree.ElementTree as ET
import xml.dom.minidom as md
import random

from progress.bar import Bar

# Choose which dataset to parse
__CHOSEN_DATASET = 5

##################
# Init Constants #
##################
__DATASETS = ['NCI1', 'PROTEINS', 'COLLAB', 'ENZYMES', 'REDDIT-BINARY', 'IMDB-BINARY']
__DATASET_NAME = __DATASETS[__CHOSEN_DATASET]
__FOLDER = f'./data/{__DATASET_NAME}/'
__EXTENSIONS = {
        'edges': f'{__DATASET_NAME}_A.txt',
        'graph_idx': f'{__DATASET_NAME}_graph_indicator.txt',
        'graph_labels': f'{__DATASET_NAME}_graph_labels.txt',
        'node_labels': f'{__DATASET_NAME}_node_labels.txt',
    }
# upper limit for train and val per class
# You have to take into account the number of classes to select the correct split size
# E.g. with 2 classes the size of train will be 2*upper_limit_test.
__SPLIT_CLASSES = {'NCI1': ((750, 750), (250, 250)),
                   'PROTEINS': ((390, 270), (130, 90)),
                   'COLLAB': ((1560, 465, 975), (520, 155, 325)),
                   'ENZYMES': ((60,)*6, (20,)*6),
                   'REDDIT-BINARY': ((600, 600), (200, 200)),
                   'IMDB-BINARY': ((300, 300), (100, 100)),
                   # 'REDDIT-MULTI-5K': ((240,)*5, (80,)*5)
                   }



def parser(folder, dataset):
    print('=' * 30)
    print(f'== Parse graphs {__DATASET_NAME} ==')
    print('=' * 30)

    # Create the graphs with the nodes and edges
    nodes_lbls_per_graph = load_nodes_per_graph_with_lbls(folder)
    edges = load_edges(folder)
    edges_per_graph = create_edges_per_graph(nodes_lbls_per_graph, edges)

    parse_graph_xml(nodes_lbls_per_graph, edges_per_graph, folder)

    # Create the splitting between the tr, va, te
    graph_lbls = load_file(folder, __EXTENSIONS['graph_labels'])
    labels_per_graph = defaultdict(list)

    classes = set()

    for idx, graph_lbl in enumerate(graph_lbls):
        graph_lbl = 0 if int(graph_lbl) < 0 else graph_lbl
        classes.add(int(graph_lbl))
        labels_per_graph[int(graph_lbl)].append((idx, int(graph_lbl)))

    split_tr, split_va = __SPLIT_CLASSES[dataset]

    random.seed(42)
    for class_ in sorted(classes):
        random.shuffle(labels_per_graph[class_])
        print(f'Num samples class {class_}: {len(labels_per_graph[class_])}')

    graphs_train, graphs_val, graphs_test = [], [], []

    for idx_cl, class_ in enumerate(sorted(classes)):
        graphs_train += labels_per_graph[class_][:split_tr[idx_cl]]
        graphs_val += labels_per_graph[class_][split_tr[idx_cl]:split_tr[idx_cl]+split_va[idx_cl]]
        graphs_test += labels_per_graph[class_][split_tr[idx_cl]+split_va[idx_cl]:]

    print(f'Size Train set: {len(graphs_train)}')
    print(f'Size Val set: {len(graphs_val)}')
    print(f'Size Test set: {len(graphs_test)}')

    parse_class_xml(graphs_train, folder, 'train')
    parse_class_xml(graphs_val, folder, 'validation')
    parse_class_xml(graphs_test, folder, 'test')

def parse_class_xml(classes, folder, name):
    graph_collection = ET.Element('GraphCollection')

    finger_prints = ET.SubElement(graph_collection, 'fingerprints')

    for idx_graph, class_ in classes:
        print_ = ET.SubElement(finger_prints, 'print')
        print_.set('file', f'molecule_{idx_graph}.gxl')
        print_.set('class', str(class_))

    b_xml = ET.tostring(graph_collection).decode()
    newxml = md.parseString(b_xml)

    folder_data = os.path.join(folder, 'data', '')
    Path(folder_data).mkdir(parents=True, exist_ok=True)

    filename = os.path.join(folder_data, f'{name}.cxl')
    with open(filename, mode='w') as f:
        f.write(newxml.toprettyxml(indent=' ', newl='\n'))


def parse_graph_xml(nodes_per_graph, edges_per_graph, folder):

    bar = Bar(f'Create Graphs', max=len(nodes_per_graph.keys()))

    for graph_idx in nodes_per_graph.keys():
        graph_name = f'molecule_{graph_idx}'
        nodes = nodes_per_graph[graph_idx]
        edges = edges_per_graph[graph_idx]
        gxl = ET.Element('gxl')

        graph = ET.SubElement(gxl, 'graph')
        graph.set('id', graph_name)
        graph.set('edgeids', 'true')
        graph.set('edgemode', 'undirected')

        mapper = {}

        for idx_node, node in enumerate(nodes):
            mapper[node.node_id] = idx_node

            node_xml = ET.SubElement(graph, 'node')
            node_xml.set('id', str(idx_node))

            attr = ET.SubElement(node_xml, 'attr')
            attr.set('name', 'lbl')

            int_ = ET.SubElement(attr, 'int')
            int_.text = str(node.node_lbl)

        for edge in edges:
            edge_xml = ET.SubElement(graph, 'edge')
            node_idx_start = mapper[edge.node_idx_start]
            node_idx_end = mapper[edge.node_idx_end]
            edge_xml.set('from', str(node_idx_start))
            edge_xml.set('to', str(node_idx_end))

        b_xml = ET.tostring(gxl).decode()
        newxml = md.parseString(b_xml)

        folder_data = os.path.join(folder, 'data', '')
        Path(folder_data).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(folder_data, f'{graph_name}.gxl')
        with open(filename, mode='w') as f:
            f.write(newxml.toprettyxml(indent=' ', newl='\n'))

        bar.next()
    bar.finish()


def create_edges_per_graph(nodes_lbls_per_graph, edges):
    graph_idx = 0
    edges_per_graph = defaultdict(list)
    for max, edge in edges:
        max_node_idx = nodes_lbls_per_graph[graph_idx][-1].node_id

        if max > max_node_idx:
            graph_idx += 1

        edges_per_graph[graph_idx].append(edge)

    return edges_per_graph


def load_edges(folder):
    raw_edges = load_file(folder, __EXTENSIONS['edges'])

    edges = []
    Edge = namedtuple('Edge', ['node_idx_start', 'node_idx_end'])

    for edge in raw_edges:
        idx_1, idx_2 = [int(val) - 1 for val in edge.split(',')]

        edges.append((max(idx_1, idx_2), Edge(idx_1, idx_2)))

    return edges


def load_nodes_per_graph_with_lbls(folder):
    graph_idx = load_file(folder, __EXTENSIONS['graph_idx'])
    node_labels = load_file(folder, __EXTENSIONS['node_labels'])

    # print(node_labels)

    if not node_labels:
        # In case of a dataset does not have node labels
        # All the labels are set to 1 (arbitrary choice).
        node_labels = ['1\n'] * len(graph_idx)

    nodes_per_graph = defaultdict(list)
    Node = namedtuple('Node', ['node_id', 'node_lbl'])

    for idx, (gr_idx, lbl) in enumerate(zip(graph_idx, node_labels)):
        node_label = int(lbl.split(',')[-1])
        nodes_per_graph[int(gr_idx)-1].append(Node(idx, node_label))

    return nodes_per_graph


def load_file(folder, extension):
    filename = os.path.join(folder, extension)
    # Check if the file exist
    # Handle the exception if a dataset doesn't have a node labels for example.
    if os.path.exists(filename):
        with open(filename, mode='r') as fp:
            data = fp.readlines()
        return data
    else:
        return []


def main():
    parser(__FOLDER, __DATASET_NAME)


if __name__ == '__main__':
    main()
