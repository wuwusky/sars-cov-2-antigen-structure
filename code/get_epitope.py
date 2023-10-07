import os
import pandas as pd
import numpy as np
import re
import zipfile
from tqdm import tqdm


# read pdb file
def read_pdb(_file):
    fp = open(_file, 'r')  # open a pdb file
    res = []
    for lne in fp.readlines():
        if lne.startswith('ATOM'):
            temp = [lne[11:17].strip(), lne[17:21].strip(), lne[21:23].strip(), lne[23:30].strip(),
                    lne[30:38].strip(), lne[38:46].strip(), lne[46:54].strip()]
            # [atom type, amino acid, chain, epitope id, x, y, z]
            res.append(temp)
    fp.close()
    return res


# identify antigen or antibody part of pdb files
def separate_antigen_antibody(_file, _chain_id):
    # make dic: {chain id : sequence} (link pdb with fasta)
    seq_dic = {}
    fp2 = open(_file, 'r')  # open a fasta file
    c_id = 0
    temp_lines = fp2.readlines()
    for lne in temp_lines:
        if not lne.startswith('>'):
            try:
                seq_dic[_chain_id[c_id]] = lne.strip('\n')
                c_id += 1
            except Exception as e:
                continue
    fp2.close()
    # identify antigen sequence & antibody sequence
    H_format = 'WG.G'  # heavy chain feature
    L_format = 'FG.G'  # light chain feature
    atb = []  # list of antibody
    atg = []  # list of antigen
    for k, v in seq_dic.items():
        if re.findall(H_format, v) or re.findall(L_format, v):
            atb.append(k)
        else:
            atg.append(k)
    return atb, atg  # antibody list, antigen list


# find target amino acid (by CA)
def get_target_aa(atb, atg, _df_pdb):
    df_atb_CA = _df_pdb.loc[np.isin(_df_pdb['chain'], atb)].loc[_df_pdb['atom type'] == 'CA']
    df_atg_CA = _df_pdb.loc[np.isin(_df_pdb['chain'], atg)].loc[_df_pdb['atom type'] == 'CA']
    selected_atb_list = []
    selected_atg_list = []
    print('Start searching target amino acid...')
    for _, row1 in df_atb_CA.iterrows():
        for _, row2 in df_atg_CA.iterrows():
            coord_atb = [float(row1['x']), float(row1['y']), float(row1['z'])]
            coord_atg = [float(row2['x']), float(row2['y']), float(row2['z'])]
            dist = np.sqrt(np.sum(np.square(np.subtract(np.array(coord_atb), np.array(coord_atg)))))
            if dist < 10:
                selected_atb_list.append(row1['epitope id'])
                selected_atg_list.append(row2['epitope id'])

    selected_atb_df = _df_pdb.loc[np.isin(_df_pdb['epitope id'], selected_atb_list)]
    selected_atg_df = _df_pdb.loc[np.isin(_df_pdb['epitope id'], selected_atg_list)]
    print('Number of target amino acid in antigen: {}, antibody: {}'
          .format(len(set(selected_atg_list)), len(set(selected_atb_list))))
    return selected_atb_df, selected_atg_df


def identify_bonds(_atb_df, _atg_df):
    hydrogen_bond = []
    salt_bridge = []
    pos_aa = ['LYS', 'ARG', 'HIS']
    neg_aa = ['ASP', 'GLU']
    # identify hydrogen bonds and salt bridges
    print('Start searching bonds...')
    for _, row1 in _atb_df.iterrows():
        for _, row2 in _atg_df.iterrows():
            coord1 = [float(row1['x']), float(row1['y']), float(row1['z'])]
            coord2 = [float(row2['x']), float(row2['y']), float(row2['z'])]
            atom1 = row1['atom type']
            atom2 = row2['atom type']
            # e.g. only count in X-H-O type in this template
            if atom1.find('O') == -1 and atom2.find('O') == -1:
                continue
            # calculate distance & identify bonds
            dist = round(np.sqrt(np.sum(np.square(np.subtract(np.array(coord1), np.array(coord2))))), 2)
            save_feature = [' '.join([row1['amino acid'], row1['epitope id'], '[%s]' % row1['atom type']]),
                            ' '.join([row2['amino acid'], row2['epitope id'], '[%s]' % row2['atom type']]),
                            dist]
            if 2 < dist < 3.5:
                if (np.isin(row1['amino acid'], pos_aa) and np.isin(row2['amino acid'], neg_aa)) or \
                        (np.isin(row1['amino acid'], neg_aa) and np.isin(row2['amino acid'], pos_aa)):
                    salt_bridge.append(save_feature)
                else:
                    hydrogen_bond.append(save_feature)
    print('Number of hydrogen bonds: {}, salt bridges: {}'
          .format(len(hydrogen_bond), len(salt_bridge)))
    print('Searching finished!')
    return hydrogen_bond, salt_bridge


def run(_file_id):
    # get_pdb_data
    data = read_pdb('./results/pdb/%s.pdb' % _file_id)
    df_pdb = pd.DataFrame(data, columns=['atom type', 'amino acid', 'chain', 'epitope id', 'x', 'y', 'z'])
    # get chain id
    chain_id = sorted(list(set(df_pdb['chain'].tolist())))
    try:
        antibody_list, antigen_list = separate_antigen_antibody('/tcdata/%s.fasta' % _file_id, chain_id)
    except Exception as e:
        antibody_list, antigen_list = separate_antigen_antibody('./tcdata/%s.fasta' % _file_id, chain_id)
    # identify bonds
    selected_antibody_df, selected_antigen_df = get_target_aa(antibody_list, antigen_list, df_pdb)
    salt_bridge, hydrogen_bond = identify_bonds(selected_antibody_df, selected_antigen_df)
    if not os.path.exists('./results/epitope'):
        os.makedirs('./results/epitope')
    pd.DataFrame(salt_bridge).to_csv('./results/epitope/%s_Sb.csv' % _file_id, index=False, header=False)
    pd.DataFrame(hydrogen_bond).to_csv('./results/epitope/%s_Hb.csv' % _file_id, index=False, header=False)
    print('Results of file %s saved!' % _file_id)


def zip_results(_input_file, _output_file):
    f = zipfile.ZipFile(_output_file, 'w', zipfile.ZIP_DEFLATED)
    for folder in _input_file:
        f.write(folder)
        for file in os.listdir(folder):
            f.write(os.path.join(folder, file))
    f.close()
    print('result.zip saved!')

def make_zip(src_dir, out_dir):
    zipf = zipfile.ZipFile(out_dir, 'w')
    pre_len = len(os.path.dirname(src_dir))
    for parent, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, arcname)
    zipf.close()

if __name__ == '__main__':
    # get file names
    file_list = os.listdir('./results/pdb')[:]
    # predict epitopes
    for item in tqdm(file_list, ncols=100):
        file_name = item.split('.')[0]
        run(file_name)
    # save results
    # zip_results(['results/'], 'result.zip')
    make_zip('./results/', '../prediction_result/result.zip')
