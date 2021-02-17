from glob import glob

def main():
    c_files = glob('**/*.c', recursive=True)
    so_files = glob('**/*.so', recursive=True)

    print(c_files)
    print(so_files)

if __name__ == '__main__':
    main()