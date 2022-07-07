# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:54:40 2022

@author: hzx
"""


import os
import glob
from multiprocessing import Pool
import pickle as pkl
import numpy as np
from Bio.PDB.PDBParser import PDBParser


parser = PDBParser()


def pdb_files(pdb_path):
    pdb_files = sorted(glob.glob(os.path.join(pdb_path,'*.pdb')))
    return pdb_files


def pdb2backbone(pdb_file):
    pdb_id = pdb_file.split('.')[0]
    structure = parser.get_structure(pdb_id, pdb_file)
    backbone = []
    for residue in structure.get_residues(): 
        atoms_per_res = []                           
        for atom in residue:
            if atom.name in ['CA', 'C', 'N', 'O']:
                atoms_per_res.append(atom.get_coord())
        if len(atoms_per_res) == 4:
            backbone.append(np.array(atoms_per_res))
    return backbone


def get_backbones(pdbs):
    pool = Pool()
    backbones = pool.map(pdb2backbone, pdbs)
    pool.close()
    pool.join()
    return backbones


def standardize(backbones, num_res=128):
    backbones_std = []
    for backbone in backbones:
        if len(backbone) >= num_res:
            backbones_std.append(backbone[ : num_res])
    return np.array(backbones_std)


def to_pkl(backbones, pkl_file='./protein_backbones.pkl'):
    with open(pkl_file,'wb') as f:
        pkl.dump(backbones, f, protocol=pkl.HIGHEST_PROTOCOL)


def load_data(pkl_file='./protein_backbones.pkl'):
    with open(pkl_file,'rb') as f:
        datas = pkl.load(f)
    return datas


if __name__ == "__main__":

    path = "../datas/LH_Combined_Chothia/"
    pdbs = pdb_files(path)
    backbones = get_backbones(pdbs)
    backbones_std = standardize(backbones)
    to_pkl(backbones_std)
    to_pkl(backbones, pkl_file='./protein_backbones_all.pkl')