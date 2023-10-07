import os
from tqdm import tqdm

# process fasta
def preprocess_fasta(_file):
    try:
        fp = open('../tcdata/%s' % _file, 'r')  # open a fasta file
    except Exception as e:
        fp = open('./tcdata/%s' % _file, 'r') 
    save_file = os.path.join(input_dir, _file)
    fp2 = open(save_file, 'w')  # rewrite fasta file intp input dir
    for lne in fp.readlines():
        if not lne.startswith('>'):
            lne.replace('X', 'G')
            if len(lne) > 999:
                lne = lne[:999] + '\n'
        fp2.write(lne)
    fp.close()
    fp2.close()


if __name__ == '__main__':
    try:
        tcdata_list = os.listdir('../tcdata/')
    except Exception as e:
        tcdata_list = os.listdir('./tcdata')

    input_dir = './input/'
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    for fa in tqdm(tcdata_list):
        preprocess_fasta(fa)
