cdef:
    ### File Extensions ###
    EXTENSION_GRAPHS = '*.gxl'
    EXTENSION_SPLITS = '.cxl'

    ### Datatset and Labels folder ###
    dict DEFAULT_FOLDERS = {'letter': './data/Letter/Letter/HIGH/',
                            'AIDS': './data/AIDS/data/',
                            'mutagenicity': './data/Mutagenicity/data/',
                            'NCI1': './data/NCI1/data/'}

    dict DEFAULT_FOLDERS_LABELS = DEFAULT_FOLDERS

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

    dict NCI1_LBL_TO_CODE = {
        '0': 0, '1': 1,
    }

    dict DEFAULT_LABELS_TO_CODE = {
        'letter': LETTER_LBL_TO_CODE,
        'AIDS': AIDS_LBL_TO_CODE,
        'mutagenicity': MUTAGENICITY_LBL_TO_CODE,
        'NCI1': NCI1_LBL_TO_CODE,
    }

def get_default_lbls_to_code():
    """Access to the DEFAULT_LABELS_TO_CODE from python."""
    return DEFAULT_LABELS_TO_CODE

def get_code_to_lbls_letter():
    return CODE_TO_LBL_LETTER

def get_code_to_lbls_muta():
    return CODE_TO_LBL_MUTAGENICITY