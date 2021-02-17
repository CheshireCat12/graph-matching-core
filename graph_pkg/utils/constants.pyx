cdef:
    ### Datatset and Labels folder ###
    dict DEFAULT_FOLDERS = {'letter': './data/Letter/Letter/HIGH/',
                            'AIDS': './data/AIDS/data/',
                            'mutagenicity': './data/Mutagenicity/data/'}

    dict DEFAULT_FOLDERS_LABELS = DEFAULT_FOLDERS

    ### Convert Labels to unique code ###
    dict LETTER_LBL_TO_CODE = {
        'A': 0, 'E': 1, 'F': 2, 'H': 3, 'I': 4, 'K': 5, 'L': 6,
        'M': 7, 'N': 8, 'T': 9, 'V': 10, 'W': 11, 'X': 12, 'Y': 13, 'Z': 14,
    }

    dict AIDS_LBL_TO_CODE = {
        'a': 0, 'i': 1,
    }

    dict MUTAGENICITY_TO_CODE = {
        'mutagen': 0, 'nonmutagen': 1,
    }

    dict DEFAULT_LABELS_TO_CODE = {
        'letter': LETTER_LBL_TO_CODE,
        'AIDS': AIDS_LBL_TO_CODE,
        'mutagenicity': MUTAGENICITY_TO_CODE,
    }