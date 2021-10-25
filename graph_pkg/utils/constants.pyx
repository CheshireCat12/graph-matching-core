cdef:
    #########################
    ##  General Constants  ##
    #########################

    ### File Extensions ###
    str EXTENSION_GRAPHS = '*.gxl'
    str EXTENSION_GRAPHML = '*.graphml'
    str EXTENSION_SPLITS = '.cxl'

    #########################
    ##  Folder Constants   ##
    #########################

    ### Datatset and Labels folder ###
    dict DEFAULT_FOLDERS = {'letter': './data/Letter/Letter/HIGH/',
                            'AIDS': './data/AIDS/data/',
                            'mutagenicity': './data/Mutagenicity/data/',
                            'NCI1': './data/NCI1/data/',
                            'proteins_tu': './data/PROTEINS/data/',
                            'enzymes': './data/ENZYMES/data/',
                            'collab': './data/COLLAB/data/',
                            'reddit_binary': './data/REDDIT-BINARY/data',
                            'IMDB_binary': './data/IMDB-Binary/data'
                            }

    dict DEFAULT_FOLDERS_GNN_EMBEDDING = {
        'enzymes': './data_gnn/reduced_graphs_ENZYMES/data/'
    }

    dict DEFAULT_FOLDERS_LABELS = DEFAULT_FOLDERS
    dict DEFAULT_FOLDERS_GNN_EMBEDDING_LABELS = DEFAULT_FOLDERS_GNN_EMBEDDING

    #################################
    ##  Convert Labels Constants   ##
    #################################

    ### Convert Labels to unique code ###
    dict LETTER_LBL_TO_CODE = {
        'A': 0, 'E': 1, 'F': 2, 'H': 3, 'I': 4, 'K': 5, 'L': 6,
        'M': 7, 'N': 8, 'T': 9, 'V': 10, 'W': 11, 'X': 12, 'Y': 13, 'Z': 14,
    }
    dict CODE_TO_LBL_LETTER = {val: key for key, val in LETTER_LBL_TO_CODE.items()}

    dict AIDS_LBL_TO_CODE = {
        'a': 0, 'i': 1,
    }

    dict MUTAGENICITY_LBL_TO_CODE = {
        'mutagen': 0, 'nonmutagen': 1,
    }
    dict CODE_TO_LBL_MUTAGENICITY = {val: key for key, val in MUTAGENICITY_LBL_TO_CODE.items()}

    dict NCI1_LBL_TO_CODE = {str(i): i for i in range(2)}

    dict PROTEINS_TU_LBL_TO_CODE = {str(i): i for i in range(1, 3)}

    dict ENZYMES_TO_CODE = {str(i): i for i in range(1, 7)}

    dict COLLAB_TO_CODE = {str(i): i for i in range(1, 4)}

    dict REDDIT_BINARY_TO_CODE = {str(i): i for i in range(2)}

    dict IMDB_BINARY_TO_CODE = {str(i): i for i in range(2)}

    dict DEFAULT_LABELS_TO_CODE = {
        'letter': LETTER_LBL_TO_CODE,
        'AIDS': AIDS_LBL_TO_CODE,
        'mutagenicity': MUTAGENICITY_LBL_TO_CODE,
        'NCI1': NCI1_LBL_TO_CODE,
        'proteins_tu': PROTEINS_TU_LBL_TO_CODE,
        'enzymes': ENZYMES_TO_CODE,
        'collab': COLLAB_TO_CODE,
        'reddit_binary': REDDIT_BINARY_TO_CODE,
        'IMDB_binary': IMDB_BINARY_TO_CODE,
    }

    ###############################
    ##  hierarchical Constants   ##
    ###############################

    list PERCENT_HIERARCHY = [1.0, 0.8, 0.6, 0.4, 0.2]


########################
##  Sigma Constants   ##
########################

THRESHOLDS_SIGMAJS = {
    'letter': 1,
    'AIDS': 1,
    'mutagenicity': 5,
    'NCI1': 5,
    'proteins_tu': 5,
    'enzymes': 5,
    'IMDB_binary': 1,
}

DATASETS_TO_ADD_EXTRA_LAYOUT = ['mutagenicity', 'NCI1', 'enzymes', 'IMDB_binary']

def get_default_lbls_to_code():
    """Access to the DEFAULT_LABELS_TO_CODE from python."""
    return DEFAULT_LABELS_TO_CODE

def get_code_to_lbls_letter():
    return CODE_TO_LBL_LETTER

def get_code_to_lbls_muta():
    return CODE_TO_LBL_MUTAGENICITY
