from glob import glob
import os

def _delete_files(files):
    for file in files:
        os.remove(file)
        print(f'File {file} Deleted successfully')

def _confirm_deletion(files, msg_confirm):
    rtrn_line = '\n'

    print(f'\n\n'
          f'!!Caution the following files are going to be deleted!!\n'
          f'Nb files to delete: {len(files)}\n\n'
          f'{rtrn_line.join(files)}\n')
    confirm = input(f'To confirm the file deletion type \'{msg_confirm}\': ')
    if confirm != msg_confirm:
        return

    _delete_files(files)

def main():
    msg_confirm_c = 'confirm.c'
    msg_confirm_so = 'confirm.so'
    delete_folders = ['graph_pkg', 'experiments']

    c_files = glob('**/*.c', recursive=True)
    so_files = glob('**/*.so', recursive=True)

    c_files_pkg = [file for file in c_files if file.split('/')[0] in delete_folders]
    so_files_pkg = [file for file in so_files if file.split('/')[0] in delete_folders]

    _confirm_deletion(c_files_pkg, msg_confirm_c)
    _confirm_deletion(so_files_pkg, msg_confirm_so)


if __name__ == '__main__':
    main()