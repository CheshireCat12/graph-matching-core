from pathlib import Path
import json


cdef class Logger:

    cdef:
        str filename
        str folder
        str lvl_name
        dict __data

    def __init__(self, str filename):
        self.filename = filename
        self.folder = '/'.join(filename.split('/')[:-1])
        Path(self.folder).mkdir(parents=True, exist_ok=True)
        self.__data = {}
        self.lvl_name = None

    property data:
        def __get__(self):
            if self.lvl_name:
                return self.__data[self.lvl_name]
            else:
                return self.__data

    def set_lvl(self, str lvl_name):
        """
        Change the level of the logger.
        It is used to log experiment with multiple loop
        :param lvl_name:
        :return:
        """
        self.lvl_name = lvl_name
        self.__data[lvl_name] = {}

    def save_data(self):
        with open(self.filename, 'w') as file:
            json.dump(self.__data, file, indent=4)