import  gc
import logging
import os
import sys
import time

import torch
import omegafold as of

# from omegafold import pipeline
import argparse
from torch.utils.hipify import hipify_python
from omegafold.pipeline import _set_precision, _get_device, fasta2inputs, save_pdb, fasta2inputs_mul
from tqdm import tqdm

def get_args_test():
    """
    Parse the arguments, which includes loading the weights

    Returns:
        input_file: the path to the FASTA file to load sequences from.
        output_dir: the output folder directory in which the PDBs will reside.
        batch_size: the batch_size of each forward
        weights: the state dict of the model

    """
    parser = argparse.ArgumentParser(
        description=
        """
        Launch OmegaFold and perform inference on the data. 
        Some examples (both the input and output files) are included in the 
        Examples folder, where each folder contains the output of each 
        available model from model1 to model3. All of the results are obtained 
        by issuing the general command with only model number chosen (1-3).
        """
    )
    # parser.add_argument(
    #     'input_file', type=lambda x: os.path.expanduser(str(x)),
    #     help=
    #     """
    #     The input fasta file
    #     """
    # )
    # parser.add_argument(
    #     'output_dir', type=lambda x: os.path.expanduser(str(x)),
    #     help=
    #     """
    #     The output directory to write the output pdb files. 
    #     If the directory does not exist, we just create it. 
    #     The output file name follows its unique identifier in the 
    #     rows of the input fasta file"
    #     """
    # )
    parser.add_argument(
        '--num_cycle', default=8, type=int,
        help="The number of cycles for optimization, default to 10"
    )
    parser.add_argument(
        '--subbatch_size', default=512, type=int,
        help=
        """
        The subbatching number, 
        the smaller, the slower, the less GRAM requirements. 
        Default is the entire length of the sequence.
        This one takes priority over the automatically determined one for 
        the sequences
        """
    )
    parser.add_argument(
        '--device', default=None, type=str,
        help=
        'The device on which the model will be running, '
        'default to the accelerator that we can find'
    )
    # parser.add_argument(
    #     '--weights_file',
    #     default='./weights/model.pt',
    #     type=str,
    #     help='The model cache to run'
    # )
    # parser.add_argument(
    #     '--weights',
    #     default="https://helixon.s3.amazonaws.com/release1.pt",
    #     type=str,
    #     help='The url to the weights of the model'
    # )
    parser.add_argument(
        '--pseudo_msa_mask_rate', default=0.12, type=float,
        help='The masking rate for generating pseudo MSAs'
    )
    parser.add_argument(
        '--num_pseudo_msa', default=10, type=int,
        help='The number of pseudo MSAs'
    )
    parser.add_argument(
        '--allow_tf32', default=True, type=hipify_python.str2bool,
        help='if allow tf32 for speed if available, default to True'
    )

    args = parser.parse_args()
    _set_precision(args.allow_tf32)

    # weights_url = args.weights
    # weights_file = args.weights_file
    # # if the output directory is not provided, we will create one alongside the
    # # input fasta file
    # if weights_file or weights_url:
    #     weights = _load_weights(weights_url, weights_file)
    #     weights = weights.pop('model', weights)
    # else:
    #     weights = None
    try:
        weights = torch.load('./user_data/weights/model.pt', map_location='cpu')
    except Exception as e:
        print(e)
        weights = torch.load('e:/model.pt', map_location='cpu')
        pass
    weights = weights.pop('model', weights)

    forward_config = argparse.Namespace(
        subbatch_size=args.subbatch_size,
        num_recycle=args.num_cycle,
    )

    args.device = _get_device(args.device)

    return args, weights, forward_config


@torch.no_grad()
def predict(input_dir, output_dir):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    args, state_dict, forward_config = get_args_test()

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'constructing omegafold')
    model = of.OmegaFold(of.make_config())
    if "model" in state_dict:
            state_dict = state_dict.pop("model")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)

    logging.info(f"Reading {input_dir}")
    input_data, save_path, ch_ids = fasta2inputs_mul(
                input_dir,
                num_pseudo_msa=args.num_pseudo_msa,
                output_dir=output_dir,
                device=args.device,
                mask_rate=args.pseudo_msa_mask_rate,
                num_cycle=args.num_cycle,
    )



    logging.info(f"Predicting all chains in {input_dir}")
    logging.info(
        f"{len(input_data[0]['p_msa'][0])} residues in this chain."
    )
    logging.info(ch_ids)
    ts = time.time()
    # try:
    output = model(
            input_data,
            predict_with_confidence=True,
            fwd_cfg=forward_config
        )
    # except RuntimeError as e:
    #     logging.info(f"Failed to generate {save_path} due to {e}")
    #     logging.info(f"Skipping...")
    #     continue
    logging.info(f"Finished prediction in {time.time() - ts:.2f} seconds.")

    logging.info(f"Saving prediction to {save_path}")
    save_pdb(
        pos14=output["final_atom_positions"],
        b_factors=output["confidence"] * 100,
        sequence=input_data[0]["p_msa"][0],
        mask=input_data[0]["p_msa_mask"][0],
        save_path=save_path,
        model=0,
        init_chain=ch_ids[0]
    )
    logging.info(f"Saved")
    del output
    torch.cuda.empty_cache()
    gc.collect()
    logging.info("Done!")

    return ch_ids

def combine_pdb(pdb_dir, save_dir):
    end_line = 'END   \n'
    list_pdb_files = os.listdir(pdb_dir)
    lines_all = ''
    for temp_pdb_name in list_pdb_files:
        temp_pdb_dir = pdb_dir + temp_pdb_name
        with open(temp_pdb_dir, mode='r') as f:
            temp_lines = f.read()
        lines_all = lines_all + temp_lines[:-len(end_line)]
    lines_all += end_line
    with open(save_dir, mode='w') as f:
        f.write(lines_all)

def analysis_cys(temp_lines, id_line):
    temp_sum_cys = 0
    for i in range(40):
        try:
            temp_type = temp_lines[id_line+i][17:20]
            if temp_type == 'CYS':
                temp_sum_cys += 1
        except Exception as e:
            continue
    if temp_sum_cys > 36:
        return True
    else:
        return False

def reconstruct_pdb(pdb_dir, save_dir, ch_ids):
    list_pdb_files = os.listdir(pdb_dir)

    temp_pdb_name = list_pdb_files[0]
    temp_pdb_dir = pdb_dir + temp_pdb_name
    with open(temp_pdb_dir, mode='r') as f:
        temp_lines = f.readlines()
    
    temp_lines_new = []
    current_ch_id = 0
    flag_ter = False
    for i, temp_line in enumerate(temp_lines):
        temp_type = temp_line[17:20]
        if 'CYS' == temp_type and flag_ter==False:
            if analysis_cys(temp_lines, i):
                flag_ter=True
                temp_lines_new.append('TER\n')
                continue
            else:
                temp_ch_name = ch_ids[current_ch_id]
                temp_line_new = temp_line.replace(' '+ch_ids[0] +' ', ' '+ temp_ch_name +' ')
                temp_lines_new.append(temp_line_new)
        elif flag_ter==True and 'CYS' == temp_type:
            continue
        elif flag_ter==True and 'CYS' != temp_type and 'END' not in temp_line:
            flag_ter=False
            current_ch_id += 1
            temp_ch_name = ch_ids[current_ch_id]
            temp_line_new = temp_line.replace(' '+ch_ids[0] +' ', ' '+ temp_ch_name +' ')
            temp_lines_new.append(temp_line_new)
        else:
            temp_ch_name = ch_ids[current_ch_id]
            temp_line_new = temp_line.replace(' '+ch_ids[0] +' ', ' '+ temp_ch_name +' ')
            temp_lines_new.append(temp_line_new)
    with open(save_dir, mode='w') as w:
        for temp_line in temp_lines_new:
            w.write(temp_line)





if __name__ == '__main__':
    input_dir = './input/'
    output_dir_temp = './pdb_temp/'
    output_dir = './results/pdb/'
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    list_files = os.listdir(input_dir)
    for temp_file in tqdm(list_files, ncols=100):
        temp_input_dir = input_dir + temp_file
        temp_name = temp_file.split('.')[0]
        temp_output_dir = output_dir_temp + temp_name
        ch_ids = predict(temp_input_dir, temp_output_dir)
        reconstruct_pdb(output_dir_temp+temp_name+'/', output_dir+temp_name+'.pdb', ch_ids)