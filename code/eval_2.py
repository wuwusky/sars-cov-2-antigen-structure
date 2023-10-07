import os
import shutil
import zipfile






def save_result(list_results_a, list_results_n):
    num_a = len(list_results_a)
    num_b = len(list_results_n)
    num_max = max(num_a, num_b)
    list_lines = []
    list_lines.append('label_a,label_n')
    for i in range(num_max):
        try:
            temp_a = list_results_a[i]
        except Exception as e:
            temp_a = ''
        try:
            temp_b = list_results_n[i]
        except Exception as e:
            temp_b = ''

        list_lines.append(str(temp_a) +','+ str(temp_b))
    
    with open('./result.csv', mode='w', encoding='gb2312', newline='') as f:
        for row in list_lines:
            f.write(row+'\n')
    print('test')

def make_zip(src_dir, out_dir):
    zipf = zipfile.ZipFile(out_dir, 'w')
    pre_len = len(os.path.dirname(src_dir))
    for parent, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, arcname)
    zipf.close()


def test_pipeline():
    test_root_dir = '../tcdata/'
    list_test_names = os.listdir(test_root_dir)
    

    # ## pdb
    # results_save_dir = './results/pdb/'
    # if os.path.exists(results_save_dir) is False:
    #     os.makedirs(results_save_dir)
    # for test_name in list_test_names:
    #     temp_result_pdb = './demo.pdb'
    #     test_name = test_name.split('.')[0]
    #     shutil.copy(temp_result_pdb, results_save_dir + test_name + '.pdb')

    ## epitope
    results_save_dir = './results/epitope/'
    if os.path.exists(results_save_dir) is False:
        os.makedirs(results_save_dir)
    for test_name in list_test_names:
        test_name = test_name.split('.')[0]
        temp_result_csv = '../user_data/demo.csv'
        shutil.copy(temp_result_csv, results_save_dir+test_name+'_Hb.csv')
        shutil.copy(temp_result_csv, results_save_dir+test_name+'_Sb.csv')
    
    make_zip('./results/', '../prediction_result/result.zip')


    
    

        







if __name__ == '__main__':
    test_pipeline()