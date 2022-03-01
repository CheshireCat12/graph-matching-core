from glob import glob
import os
os.system('')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\x1b[0;31;40m'
    WARNING = '\33[1;31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _delete_files(files):
    for file in files:
        os.remove(file)
        print(f'File {file} Deleted successfully')

def _confirm_deletion(files, msg_confirm):
    rtrn_line = '\n\t'
    msg_warning = '    !!Caution the following files are going to be deleted!!'
    nb_equal = len(msg_warning) + 4
    print(f'\n\n{msg_warning}\n'
          f'{"="*nb_equal}\n'
          f'{bcolors.WARNING}\t{rtrn_line.join(files)}{bcolors.ENDC}\n')

    print(f'Summary\n'
          f'{"="*nb_equal}\n'
          f'    Nb files to delete: {len(files)}\n')
    confirm = input(f'To confirm the file deletion type \'{msg_confirm}\': ')
    if confirm != msg_confirm:
        return

    _delete_files(files)

def main():
    msg_confirm_c = 'confirm.c'
    msg_confirm_so = 'confirm.so'
    delete_folders = ['graph_pkg_core']

    c_files = glob('**/*.c', recursive=True)
    so_files = glob('**/*.so', recursive=True)

    c_files_pkg = [file for file in c_files if file.split('/')[0] in delete_folders]
    so_files_pkg = [file for file in so_files if file.split('/')[0] in delete_folders]

    _confirm_deletion(c_files_pkg, msg_confirm_c)
    _confirm_deletion(so_files_pkg, msg_confirm_so)


if __name__ == '__main__':
    main()