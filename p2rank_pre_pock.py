import io
import os
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            a = res.resname
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def clean_protein(path, clean_path):
    os.system(f"mkdir -p {clean_path}")
    files = os.listdir(path)
    for file in files:
        fi = file.split('.')[0]
        fi_path = os.path.join(path, file)
        parser = PDBParser(QUIET=True)
        s = parser.get_structure("x", fi_path)
        all_res = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True) 
        toFile = f"{clean_path}/{fi}_protein.pdb"
        io=PDBIO()
        io.set_structure(s)
        io.save(toFile)


def p2rank_pred(p2rank_prediction_folder):
    os.system(f"mkdir -p {p2rank_prediction_folder}")
    os.system(f"mkdir -p {p2rank_prediction_folder}/p2rank")
    ds = f"{p2rank_prediction_folder}/protein_list.ds"
    with open(ds, "w") as out:
        clean_path = './pdbfile/kinase/alpha_davis/clean_pdbs'
        files = os.listdir(clean_path)
        for file in files:
            out.write(f"../clean_pdbs/{file}\n")

    p2rank = "bash ./p2rank_2.3/prank"
    cmd = f"{p2rank} predict {ds} -o {p2rank_prediction_folder}/p2rank -threads 16"
    os.system(cmd)
