from xmltodict import parse
from glob import glob

def convert_letter_gxl_to_graphml(graph_files):

    for graph_file in graph_files:
        with open(graph_file) as file:
            lines = file.readlines()
            graph = parse("".join(lines))
            graph_idx = graph['gxl']['graph']['@id']
            graph_edge_mode = graph['gxl']['graph']['@edgemode']
            nodes = graph['gxl']['graph']['node']
            edges = graph['gxl']['graph']['edge']

            print(graph['gxl']['graph'])


        break


if __name__ == '__main__':
    graph_files = glob('./data/Letter/Letter/LOW/*.gxl')
    # print(graph_files)
    convert_letter_gxl_to_graphml(graph_files)