import os.path
from tqdm import tqdm
import joblib
import pandas as pd
from rdkit import Chem
import sys
import copy
import networkx as nx
import argparse

import multiprocessing
from collections import Counter

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cur_path, '../'))
from Codes.mol_info import Label_Vocab, BOND_TYPES, BOND_FLOAT_TO_TYPE, SINGLE_ATTACH
from Codes.mol_info import MAX_VALENCE


class RxnElement:
    """
    RxnElement is an abstract class for dealing with single molecule. The graph
    and corresponding molecule attributes are built for the molecule. The constructor
    accepts only mol objects, sidestepping the use of SMILES string which may always
    not be achievable, especially for a unkekulizable molecule.
    """

    def __init__(self, mol: Chem.Mol, rxn_class: int = None) -> None:
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        """
        self.mol = mol
        self.rxn_class = rxn_class
        self._build_mol()
        self._build_graph()

    def _build_mol(self) -> None:
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms()
        self.num_bonds = self.mol.GetNumBonds()
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                            for atom in self.mol.GetAtoms()}
        self.idx_to_amap = {value: key for key, value in self.amap_to_idx.items()}

    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index(bond.GetBondType())
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        self.atom_scope = (0, self.num_atoms)
        self.bond_scope = (0, self.num_bonds)

    # CHECK IF THESE TWO ARE NEEDED
    def update_atom_scope(self, offset: int):
        """Updates the atom indices side the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        if isinstance(self.atom_scope, list):
            return [(st + offset, le) for (st, le) in self.atom_scope]
        st, le = self.atom_scope
        return (st + offset, le)

    def update_bond_scope(self, offset: int):
        """Updates the atom indices side the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        if isinstance(self.bond_scope, list):
            return [(st + offset, le) for (st, le) in self.bond_scope]
        st, le = self.bond_scope
        return (st + offset, le)


class MultiElement(RxnElement):
    """
    MultiElement is an abstract class for dealing with multiple molecules. The graph
    is built with all molecules, but different molecules and their sizes are stored.
    The constructor accepts only mol objects, sidestepping the use of SMILES string
    which may always not be achievable, especially for an invalid intermediates.
    """

    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index(bond.GetBondType())
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        frag_indices = [c for c in nx.strongly_connected_components(self.G_dir)]
        self.mols = [get_sub_mol(self.mol, sub_atoms) for sub_atoms in frag_indices]

        atom_start = 0
        bond_start = 0
        self.atom_scope = []
        self.bond_scope = []

        for mol in self.mols:
            self.atom_scope.append((atom_start, mol.GetNumAtoms()))
            self.bond_scope.append((bond_start, mol.GetNumBonds()))
            atom_start += mol.GetNumAtoms()
            bond_start += mol.GetNumBonds()


def generate_og_to_new_dict():
    if not os.path.exists(r'og_to_new.dict'):
        og_to_new = {}
        new = []
        if not os.path.exists(r'new_side_smiles.txt'):
            print('Please rerun after downloading or modifying the original side product text(og_side_smiles.txt) '
                  'and rename it as new_side_smiles.txt ')
            exit(0)
        with open(r'new_side_smiles.txt', 'r+') as r:
            for line in r.readlines():
                new.append(line.strip())
        og = []
        with open(r'og_side_smiles.txt', 'r+') as r:
            for line in r.readlines():
                og.append(line.strip())
        for i, j in zip(og, new):
            og_to_new[i] = j
        joblib.dump(og_to_new, 'og_to_new.dict')
    else:
        og_to_new = joblib.load('og_to_new.dict')
    return og_to_new


def get_mol(smiles: str, kekulize: bool = False) -> Chem.Mol:
    """SMILES string to Mol.

    Parameters
    ----------
    smiles: str,
        SMILES string for molecule
    kekulize: bool,
        Whether to kekulize the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None and kekulize:
        Chem.Kekulize(mol)
    return mol


def get_sub_mol(mol: Chem.Mol, sub_atoms) -> Chem.Mol:
    """Extract subgraph from molecular graph.

    Parameters
    ----------
    mol: Chem.Mol,
        RDKit mol object,
    sub_atoms: List[int],
        List of atom indices in the subgraph.
    """
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx():  # each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()


def apply_edits_to_mol(mol: Chem.Mol, edits) -> Chem.Mol:
    """Apply edits to molecular graph.

    Parameters
    ----------
    mol: Chem.Mol,
        RDKit mol object
    edits: Iterable[str],
        Iterable of edits to apply. An edit is structured as a1:a2:b1:b2, where
        a1, a2 are atom maps of participating atoms and b1, b2 are previous and
        new bond orders. When  a2 = 0, we update the hydrogen count.
    """
    new_mol = Chem.RWMol(mol)
    amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()}

    # Keep track of aromatic nitrogens, might cause explicit hydrogen issues
    aromatic_nitrogen_idx = set()
    aromatic_carbonyl_adj_to_aromatic_nH = {}
    aromatic_carbondeg3_adj_to_aromatic_nH0 = {}
    for a in new_mol.GetAtoms():
        if a.GetIsAromatic() and a.GetSymbol() == 'N':
            aromatic_nitrogen_idx.add(a.GetIdx())
            for nbr in a.GetNeighbors():
                nbr_is_carbon = (nbr.GetSymbol() == 'C')
                nbr_is_aromatic = nbr.GetIsAromatic()
                nbr_has_double_bond = any(b.GetBondTypeAsDouble() == 2 for b in nbr.GetBonds())
                nbr_has_3_bonds = (len(nbr.GetBonds()) == 3)

                if (a.GetNumExplicitHs() == 1 and nbr_is_carbon and nbr_is_aromatic
                        and nbr_has_double_bond):
                    aromatic_carbonyl_adj_to_aromatic_nH[nbr.GetIdx()] = a.GetIdx()
                elif (a.GetNumExplicitHs() == 0 and nbr_is_carbon and nbr_is_aromatic
                      and nbr_has_3_bonds):
                    aromatic_carbondeg3_adj_to_aromatic_nH0[nbr.GetIdx()] = a.GetIdx()
        else:
            a.SetNumExplicitHs(0)
    new_mol.UpdatePropertyCache()

    # Apply the edits as predicted
    for edit in edits:
        x, y, prev_bo, new_bo = edit.split(":")
        x, y = int(x), int(y)
        new_bo = float(new_bo)

        if y == 0:
            continue

        bond = new_mol.GetBondBetweenAtoms(amap[x], amap[y])
        a1 = new_mol.GetAtomWithIdx(amap[x])
        a2 = new_mol.GetAtomWithIdx(amap[y])

        if bond is not None:
            new_mol.RemoveBond(amap[x], amap[y])

            # Are we losing a bond on an aromatic nitrogen?
            if bond.GetBondTypeAsDouble() == 1.0:
                if amap[x] in aromatic_nitrogen_idx:
                    if a1.GetTotalNumHs() == 0:
                        a1.SetNumExplicitHs(1)
                    elif a1.GetFormalCharge() == 1:
                        a1.SetFormalCharge(0)
                elif amap[y] in aromatic_nitrogen_idx:
                    if a2.GetTotalNumHs() == 0:
                        a2.SetNumExplicitHs(1)
                    elif a2.GetFormalCharge() == 1:
                        a2.SetFormalCharge(0)

            # Are we losing a c=O bond on an aromatic ring? If so, remove H from adjacent nH if appropriate
            if bond.GetBondTypeAsDouble() == 2.0:
                if amap[x] in aromatic_carbonyl_adj_to_aromatic_nH:
                    new_mol.GetAtomWithIdx(aromatic_carbonyl_adj_to_aromatic_nH[amap[x]]).SetNumExplicitHs(0)
                elif amap[y] in aromatic_carbonyl_adj_to_aromatic_nH:
                    new_mol.GetAtomWithIdx(aromatic_carbonyl_adj_to_aromatic_nH[amap[y]]).SetNumExplicitHs(0)

        if new_bo > 0:
            new_mol.AddBond(amap[x], amap[y], BOND_FLOAT_TO_TYPE[new_bo])

            # Special alkylation case?
            if new_bo == 1:
                if amap[x] in aromatic_nitrogen_idx:
                    if a1.GetTotalNumHs() == 1:
                        a1.SetNumExplicitHs(0)
                    else:
                        a1.SetFormalCharge(1)
                elif amap[y] in aromatic_nitrogen_idx:
                    if a2.GetTotalNumHs() == 1:
                        a2.SetNumExplicitHs(0)
                    else:
                        a2.SetFormalCharge(1)

            # Are we getting a c=O bond on an aromatic ring? If so, add H to adjacent nH0 if appropriate
            if new_bo == 2:
                if amap[x] in aromatic_carbondeg3_adj_to_aromatic_nH0:
                    new_mol.GetAtomWithIdx(aromatic_carbondeg3_adj_to_aromatic_nH0[amap[x]]).SetNumExplicitHs(1)
                elif amap[y] in aromatic_carbondeg3_adj_to_aromatic_nH0:
                    new_mol.GetAtomWithIdx(aromatic_carbondeg3_adj_to_aromatic_nH0[amap[y]]).SetNumExplicitHs(1)

    pred_mol = new_mol.GetMol()

    # Clear formal charges to make molecules valid
    # Note: because S and P (among others) can change valence, be more flexible
    for atom in pred_mol.GetAtoms():
        if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1:  # exclude negatively-charged azide
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals <= 3:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'N' and atom.GetFormalCharge() == -1:  # handle negatively-charged azide addition
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 3 and any([nbr.GetSymbol() == 'N' for nbr in atom.GetNeighbors()]):
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'N':
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 4 and not atom.GetIsAromatic():  # and atom.IsInRingSize(5)):
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == 'C' and atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'O' and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]) + atom.GetNumExplicitHs()
            if bond_vals == 2:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() in ['Cl', 'Br', 'I', 'F'] and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 1:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'S' and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals in [2, 4, 6]:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'P':  # quartenary phosphorous should be pos. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(1)
                atom.SetNumExplicitHs(0)
            elif sum(bond_vals) == 3 and len(bond_vals) == 3:  # make sure neutral
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'B':  # quartenary boron should be neg. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)
        elif atom.GetSymbol() in ['Mg', 'Zn']:
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 1 and len(bond_vals) == 1:
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == 'Si':
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == len(bond_vals):
                atom.SetNumExplicitHs(max(0, 4 - len(bond_vals)))

    return pred_mol


def canonicalize(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
    except:
        print('no mol', flush=True)
        return smiles
    if tmp is None:
        return smiles
    tmp = Chem.RemoveHs(tmp)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return Chem.MolToSmiles(tmp)


def attach_groups(prod_smi, lg_groups, lg_groups_map, attach_info):
    try:
        tmp_mol = Chem.MolFromSmiles(prod_smi)
        aromatic_co_adj_n = set()
        aromatic_co = set()
        aromatic_cs = set()
        aromatic_cn = set()

        for atom in tmp_mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                nei_symbols = [nei.GetSymbol() for nei in atom.GetNeighbors()]
                if 'O' in nei_symbols and 'N' in nei_symbols and len(atom.GetBonds()) == 3 and atom.GetIsAromatic():
                    aromatic_co_adj_n.add(atom.GetIdx())

                elif 'O' in nei_symbols and len(atom.GetBonds()) == 3 and atom.GetIsAromatic():
                    aromatic_co.add(atom.GetIdx())

                elif 'N' in nei_symbols and len(atom.GetBonds()) == 3 and atom.GetIsAromatic():
                    aromatic_cn.add(atom.GetIdx())

                elif 'S' in nei_symbols and len(atom.GetBonds()) == 3 and atom.GetIsAromatic():
                    aromatic_cs.add(atom.GetIdx())

        combined_mol = Chem.Mol()

        lg_mols_map = []
        lg_start = 1000
        bt = []
        if ('[Cl:1]' in lg_groups and 'O=[C:1]c1ccccc1[C:1]=O' in lg_groups) or (
                '[Br:1].[Br:1]' in lg_groups and '[O:1]' in lg_groups):
            bt.append('None')
            for lg_group_map in lg_groups_map:
                lg_mol_map = Chem.MolFromSmiles(lg_group_map)
                for atom in lg_mol_map.GetAtoms():
                    if atom.GetAtomMapNum() == 999:
                        atom.SetNumExplicitHs(
                            atom.GetNumExplicitHs() + 1) if atom.GetSymbol() is not 'O' else atom.SetNumExplicitHs(
                            atom.GetNumExplicitHs() + 2)
                        atom.SetAtomMapNum(0)
                combined_mol = Chem.CombineMols(combined_mol, lg_mol_map)
        else:
            for idx, lg_group in enumerate(lg_groups):
                if lg_group != "<eos>" and lg_group != 'c1c[n:1]cn1.c1c[n:1]cn1':
                    lg_mol_map = Chem.MolFromSmiles(lg_groups_map[idx])
                    bt.append(Chem.BondType.SINGLE if lg_groups[idx] in SINGLE_ATTACH else Chem.BondType.DOUBLE)
                    for atom in lg_mol_map.GetAtoms():
                        if atom.GetAtomMapNum() == 999:
                            atom.SetAtomMapNum(lg_start)
                            lg_start += 1

                    combined_mol = Chem.CombineMols(combined_mol, lg_mol_map)
                    lg_mols_map.append(lg_mol_map)

                elif lg_group == 'c1c[n:1]cn1.c1c[n:1]cn1':
                    lg_group_map = lg_groups_map[idx].replace('[n:999]', '[N:999]')
                    lg_mol_map = Chem.MolFromSmiles(lg_group_map)
                    bt.append('special smiles')
                    for atom in lg_mol_map.GetAtoms():
                        if atom.GetAtomMapNum() == 999:
                            atom.SetAtomMapNum(lg_start)
                            lg_start += 1

                    combined_mol = Chem.CombineMols(combined_mol, lg_mol_map)
                    lg_mols_map.append(lg_mol_map)
                else:
                    lg_mols_map.append("<eos>")

        bt = list(set(bt))[0]
        rw_mol = Chem.RWMol(Chem.Mol(combined_mol))
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()
                    if atom.GetAtomMapNum() != 0}
        if bt is 'special smiles':
            atom_idx = []
            for atom in rw_mol.GetAtoms():
                if atom.GetAtomMapNum() >= 1000:
                    atom_idx.append(amap_idx[atom.GetAtomMapNum()])
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            assert len(atom_idx) == 2

        elif bt == 'None':
            pass
        else:
            if len(lg_mols_map) == 2 and '<eos>' in lg_mols_map and bt is Chem.BondType.SINGLE:
                for lg_mol_map in lg_mols_map:
                    if lg_mol_map == '<eos>':
                        continue
                    else:
                        for atom in rw_mol.GetAtoms():
                            if atom.GetAtomMapNum() >= 1000:
                                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            elif len(lg_mols_map) == 2 and '<eos>' in lg_mols_map and bt is Chem.BondType.DOUBLE:
                for lg_mol_map in lg_mols_map:
                    if lg_mol_map == '<eos>':
                        continue
                    else:
                        for atom in rw_mol.GetAtoms():
                            if atom.GetAtomMapNum() >= 1000:
                                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 2)
            elif len(lg_mols_map) == 2:
                atom_idx = []
                for atom in rw_mol.GetAtoms():
                    if atom.GetAtomMapNum() >= 1000:
                        atom_idx.append(amap_idx[atom.GetAtomMapNum()])
                assert len(atom_idx) == 2
                rw_mol.AddBond(atom_idx[0], atom_idx[1], bt)

        for atom in rw_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000:
                if len(attach_info) == 1:
                    atom.SetAtomMapNum(list(attach_info.values())[0])
                elif len(attach_info) == 2:
                    if len(set(list(attach_info.keys()))) == 1:
                        if atom.GetSymbol() == list(attach_info.keys())[0]:
                            atom.SetAtomMapNum(list(attach_info.values())[0])
                            attach_info[list(attach_info.keys())[0]] = list(attach_info.values())[1]
                    elif len(set(list(attach_info.keys()))) == 2:
                        if atom.GetSymbol() == list(attach_info.keys())[0]:
                            atom.SetAtomMapNum(list(attach_info.values())[0])
                        elif atom.GetSymbol() == list(attach_info.keys())[1]:
                            atom.SetAtomMapNum(list(attach_info.values())[1])
                        else:
                            raise ValueError("i can`t believe it")
                else:
                    a = 1
                    print('error!!!!')
                    raise ValueError("error !!!!!!")

        side_prod_mol = rw_mol.GetMol()
        for atom in side_prod_mol.GetAtoms():
            if atom.GetSymbol() == 'N':
                if not atom.GetIsAromatic():
                    bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                    if bond_vals >= MAX_VALENCE['N']:
                        atom.SetNumExplicitHs(0)
                        atom.SetFormalCharge(int(bond_vals - MAX_VALENCE['N']))

                elif atom.GetIsAromatic() and atom.GetFormalCharge() == 1:
                    bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                    if bond_vals == MAX_VALENCE['N']:
                        atom.SetNumExplicitHs(0)
                        atom.SetFormalCharge(0)

            elif atom.GetSymbol() == 'C':
                check1 = atom.GetIdx() in aromatic_co_adj_n
                check2 = atom.GetIdx() in aromatic_co
                check3 = atom.GetIdx() in aromatic_cs
                check4 = atom.GetIdx() in aromatic_cn

                if check1 or check2 or check3 or check4:
                    bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                    if bond_vals >= MAX_VALENCE['C']:
                        atom.SetNumExplicitHs(0)

                else:
                    bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                    if bond_vals >= MAX_VALENCE['C']:
                        atom.SetNumExplicitHs(0)
                        atom.SetFormalCharge(int(bond_vals - MAX_VALENCE['C']))

                    elif bond_vals < MAX_VALENCE['C']:
                        atom.SetNumExplicitHs(int(MAX_VALENCE['C'] - bond_vals))
                        atom.SetFormalCharge(0)

            elif atom.GetSymbol() == 'S':
                bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                if bond_vals in [2, 4, 6]:
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(0)

            elif atom.GetSymbol() == 'Sn':
                bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                if bond_vals >= 4:
                    atom.SetNumExplicitHs(0)

            elif atom.GetSymbol() == 'O':
                bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                if bond_vals >= MAX_VALENCE['O']:
                    atom.SetNumExplicitHs(0)

                elif bond_vals < MAX_VALENCE['O'] and atom.GetFormalCharge() != -1:
                    atom.SetNumExplicitHs(int(MAX_VALENCE['O'] - bond_vals))
                    atom.SetFormalCharge(0)

            elif atom.GetSymbol() == 'B':  # quartenary boron should be neg. charge with 0 H
                bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
                if sum(bond_vals) == 4 and len(bond_vals) == 4:
                    atom.SetFormalCharge(-1)
                    atom.SetNumExplicitHs(0)

                elif sum(bond_vals) >= 3:
                    atom.SetNumExplicitHs(0)

            elif atom.GetSymbol() in ['Br', 'Cl', 'I', 'F']:
                bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                if bond_vals >= MAX_VALENCE[atom.GetSymbol()]:
                    atom.SetNumExplicitHs(0)

        side_prod_smi = Chem.MolToSmiles(side_prod_mol)
        cano_smi = canonicalize(side_prod_smi)
        return cano_smi
    except:
        return '', ''


def get_mol_list(p, r, core_edits, idx=0):
    mol_list = []
    products = get_mol(p)
    if (products is None) or (products.GetNumAtoms() <= 1):
        print(f"Product has 0 or 1 atoms, Skipping reaction {idx}")
        print()
        sys.stdout.flush()
        # continue

    reactants = get_mol(r)

    if (reactants is None) or (reactants.GetNumAtoms() <= 1):
        print(f"Reactant has 0 or 1 atoms, Skipping reaction {idx}")
        print()
        sys.stdout.flush()
        # continue

    fragments = apply_edits_to_mol(Chem.Mol(products), core_edits)
    # counter.append(len(reaction_info.core_edits))

    if len(Chem.rdmolops.GetMolFrags(fragments)) != len(Chem.rdmolops.GetMolFrags(reactants)):
        print(f"Number of fragments don't match reactants. Skipping reaction {idx}")
        print()
        sys.stdout.flush()
        # continue

    frag_mols = copy.deepcopy(MultiElement(mol=Chem.Mol(fragments)).mols)
    reac_mols = copy.deepcopy(MultiElement(mol=Chem.Mol(reactants)).mols)
    mol_list.append((products, copy.deepcopy(reac_mols), copy.deepcopy(frag_mols)))

    return mol_list


def get_bond_info(mol: Chem.Mol):
    """Get information on bonds in the molecule.

    Parameters
    ----------
    mol: Chem.Mol
        Molecule
    """
    if mol is None:
        return {}

    bond_info = {}
    for bond in mol.GetBonds():
        a_start = bond.GetBeginAtom().GetAtomMapNum()
        a_end = bond.GetEndAtom().GetAtomMapNum()

        key_pair = sorted([a_start, a_end])
        bond_info[tuple(key_pair)] = [bond.GetBondTypeAsDouble(), bond.GetIdx()]

    return bond_info


def align_kekule_pairs(r: str, p: str):
    """Aligns kekule pairs to ensure unchanged bonds have same bond order in
    previously aromatic rings.

    Parameters
    ----------
    r: str,
        SMILES string representing the reactants
    p: str,
        SMILES string representing the product
    """
    reac_mol = Chem.MolFromSmiles(r)
    max_amap = max([atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()])
    for atom in reac_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap = max_amap + 1

    prod_mol = Chem.MolFromSmiles(p)

    prod_prev = get_bond_info(prod_mol)
    Chem.Kekulize(prod_mol)
    prod_new = get_bond_info(prod_mol)

    reac_prev = get_bond_info(reac_mol)
    Chem.Kekulize(reac_mol)
    reac_new = get_bond_info(reac_mol)

    for bond in prod_new:
        if bond in reac_new and (prod_prev[bond][0] == reac_prev[bond][0]):
            reac_new[bond][0] = prod_new[bond][0]

    reac_mol = Chem.RWMol(reac_mol)
    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}

    for bond in reac_new:
        idx1, idx2 = amap_idx[bond[0]], amap_idx[bond[1]]
        bo = reac_new[bond][0]
        reac_mol.RemoveBond(idx1, idx2)
        reac_mol.AddBond(idx1, idx2, BOND_FLOAT_TO_TYPE[bo])

    return reac_mol.GetMol(), prod_mol


def map_reac_and_frag(reac_mols, frag_mols):
    """Aligns reactant and fragment mols side computing atom map overlaps.

    Parameters
    ----------
    reac_mols: List[Chem.Mol],
        List of reactant mols
    frag_mols: List[Chem.Mol],
        List of fragment mols
    """
    if len(reac_mols) != len(frag_mols):
        return reac_mols, frag_mols
    reac_maps = [[atom.GetAtomMapNum() for atom in mol.GetAtoms()] for mol in reac_mols]
    frag_maps = [[atom.GetAtomMapNum() for atom in mol.GetAtoms()] for mol in frag_mols]

    overlaps = {i: [] for i in range(len(frag_mols))}
    for i, fmap in enumerate(frag_maps):
        overlaps[i].extend([len(set(fmap).intersection(set(rmap))) for rmap in reac_maps])
        overlaps[i] = overlaps[i].index(max(overlaps[i]))

    new_frag = [Chem.Mol(mol) for mol in frag_mols]
    new_reac = [Chem.Mol(reac_mols[overlaps[i]]) for i in overlaps]
    return new_reac, new_frag


def extract_leaving_groups_for_sideprod(mol_list):
    """Extracts leaving groups from a product-fragment-reactant tuple.

    Parameters
    ----------
    mol_list: List[Tuple[Chem.Mol]]
        List of product-fragment-reactant tuples
    """
    for mol_tuple in mol_list:
        p_mol, reac_mols, frag_mols = mol_tuple

        reac_mols, frag_mols = map_reac_and_frag(reac_mols, frag_mols)

        r_mol = Chem.Mol()
        for mol in reac_mols:
            r_mol = Chem.CombineMols(r_mol, Chem.Mol(mol))

        for atom in p_mol.GetAtoms():
            atom.SetNumExplicitHs(0)

        p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in p_mol.GetAtoms()}
        r_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in r_mol.GetAtoms()}

        labels = []
        labels_map = []
        attach_info = {}
        for i, mol in enumerate(reac_mols):
            idxs = []
            attach_amaps = []

            for atom in mol.GetAtoms():
                amap = atom.GetAtomMapNum()
                if amap not in p_amap_idx and amap in r_amap_idx:
                    idxs.append(r_amap_idx[amap])
                    nei_amaps = [nei.GetAtomMapNum() for nei in atom.GetNeighbors()]
                    if any(prod_map in nei_amaps for prod_map in p_amap_idx):
                        attach_amaps.append(amap)
                        attach_info[atom.GetSymbol()] = amap

            if len(idxs):
                lg_mol = get_sub_mol(r_mol, idxs)
                lg_mol_map = copy.deepcopy(lg_mol)
                for atom in lg_mol.GetAtoms():
                    if atom.GetAtomMapNum() in attach_amaps:
                        atom.SetAtomMapNum(1)
                    else:
                        atom.SetAtomMapNum(0)

                for atom in lg_mol_map.GetAtoms():
                    if atom.GetAtomMapNum() in attach_amaps:
                        atom.SetAtomMapNum(999)

                lg = Chem.MolToSmiles(lg_mol)
                lg_map = Chem.MolToSmiles(lg_mol_map)

                labels.append(lg)
                labels_map.append(lg_map)
            else:
                labels.append('<eos>')
                labels_map.append('<eos>')

    return labels, labels_map, attach_info


def get_reaction_core(r: str, p: str, kekulize: bool = False, use_h_labels: bool = False):
    """Get the reaction core and edits for given reaction

    Parameters
    ----------
    r: str,
        SMILES string representing the reactants
    p: str,
        SMILES string representing the product
    kekulize: bool,
        Whether to kekulize molecules to fetch minimal set of edits
    use_h_labels: bool,
        Whether to use change in hydrogen counts in edits
    """
    reac_mol = get_mol(r)
    prod_mol = get_mol(p)

    if reac_mol is None or prod_mol is None:
        return set(), []

    if kekulize:
        reac_mol, prod_mol = align_kekule_pairs(r, p)

    prod_bonds = get_bond_info(prod_mol)
    p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in prod_mol.GetAtoms()}

    max_amap = max([atom.GetAtomMapNum() for atom in reac_mol.GetAtoms()])
    for atom in reac_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(max_amap + 1)
            max_amap += 1

    reac_bonds = get_bond_info(reac_mol)
    reac_amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}

    rxn_core = set()
    core_edits = []

    for bond in prod_bonds:
        if bond in reac_bonds and prod_bonds[bond][0] != reac_bonds[bond][0]:
            a_start, a_end = bond
            prod_bo, reac_bo = prod_bonds[bond][0], reac_bonds[bond][0]

            a_start, a_end = sorted([a_start, a_end])
            edit = f"{a_start}:{a_end}:{prod_bo}:{reac_bo}"
            core_edits.append(edit)
            rxn_core.update([a_start, a_end])

        if bond not in reac_bonds:
            a_start, a_end = bond
            reac_bo = 0.0
            prod_bo = prod_bonds[bond][0]

            start, end = sorted([a_start, a_end])
            edit = f"{a_start}:{a_end}:{prod_bo}:{reac_bo}"
            core_edits.append(edit)
            rxn_core.update([a_start, a_end])

    for bond in reac_bonds:
        if bond not in prod_bonds:
            amap1, amap2 = bond

            if (amap1 in p_amap_idx) and (amap2 in p_amap_idx):
                a_start, a_end = sorted([amap1, amap2])
                reac_bo = reac_bonds[bond][0]
                edit = f"{a_start}:{a_end}:{0.0}:{reac_bo}"
                core_edits.append(edit)
                rxn_core.update([a_start, a_end])

    if use_h_labels:
        if len(rxn_core) == 0:
            for atom in prod_mol.GetAtoms():
                amap_num = atom.GetAtomMapNum()

                numHs_prod = atom.GetTotalNumHs()
                numHs_reac = reac_mol.GetAtomWithIdx(reac_amap[amap_num]).GetTotalNumHs()

                if numHs_prod != numHs_reac:
                    edit = f"{amap_num}:{0}:{1.0}:{0.0}"
                    core_edits.append(edit)
                    rxn_core.add(amap_num)

    return rxn_core, core_edits


def generate_sideprod_smiles(rxn_smiles):
    r, p = rxn_smiles.split('>>')
    rxn_core, core_edits = get_reaction_core(r, p, kekulize=True, use_h_labels=True)
    mol_list = get_mol_list(p, r, core_edits)
    lg_groups, lg_groups_map, attach_info = extract_leaving_groups_for_sideprod(mol_list)
    if list(set(lg_groups)) == ['<eos>']:
        return '<eos>'
    cano_side_smiles = attach_groups(p, lg_groups, lg_groups_map, attach_info)
    return cano_side_smiles


def multi_process(data):
    global csv
    idx = data['idx']
    rxn = data['rxn']
    rxn_class = data['class']
    cano_side_smiles = generate_sideprod_smiles(rxn)

    results = {
        "idx": idx,
        "cano_side_smiles": cano_side_smiles,
        "new_class": rxn_class,
    }
    return results


def descend_cano_smiles(cano_smiles_list):
    results = Counter(cano_smiles_list)
    desced_smiles_list = sorted(results, key=results.get, reverse=True)
    return desced_smiles_list

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, default="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    os.makedirs(os.path.join(os.getcwd(), '../%s/dataset/stage_one' % args.exp_id), exist_ok=True)
    os.chdir(os.path.join(os.getcwd(), '../%s/dataset/stage_one' % args.exp_id))
    data_set = ['train', 'eval', 'test', ]
    og_side_smiles_file = r'og_side_smiles.txt'
    descend_side_file = r'descend_side_smiles.txt'
    side_label_dict_file = 'side_smiles_to_label.dict'

    for data_name in data_set:
        if not os.path.exists('%s/side_product_rxn_%s.csv' % (data_name, data_name)):
            data_path = r'%s/canonicalized_%s.csv' % (data_name, data_name)
            csv = pd.read_csv(data_path)
            rxn_list = csv['reactants>reagents>production'].tolist()
            class_list = csv['class'].tolist()
            data = [{
                'idx': idx,
                'rxn': rxn,
                'class': rxn_class
            } for idx, rxn, rxn_class in zip(list(range(len(rxn_list))), rxn_list, class_list)]
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = list(tqdm(pool.imap(multi_process, data), total=len(data)))
            pool.close()
            pool.join()
            cano_side_smiles_list = []
            new_rxn_list = []
            new_class_list = []
            for result in results:
                cano_side_smiles_list.append(result['cano_side_smiles'])
                # new_rxn_list.append(result['new_rxn'])
                new_class_list.append(result['new_class'])

            df = pd.DataFrame()
            assert len(new_class_list) == len(cano_side_smiles_list) == len(rxn_list)
            df['reactants>reagents>production'] = rxn_list

            df['class'] = new_class_list
            df['product'] = [rxn.split('>>')[1] for rxn in rxn_list]

            if data_name == 'train' and not os.path.exists(side_label_dict_file):
                descend_og_labels_list = descend_cano_smiles(cano_side_smiles_list)
                assert '' not in descend_og_labels_list

                if not os.path.exists(descend_side_file):
                    with open(og_side_smiles_file, 'w+') as n:
                        for i in descend_og_labels_list:
                            n.writelines(i)
                            n.write('\n')
                    og_to_new = generate_og_to_new_dict()
                    new_side_smiles_list = [og_to_new.get(i) for i in cano_side_smiles_list]
                    descend_new_labels_list = descend_cano_smiles(new_side_smiles_list)
                    with open(descend_side_file, 'w+') as n:
                        for i in descend_new_labels_list:
                            n.writelines(i)
                            n.write('\n')
                else:
                    descend_new_labels_list = []
                    with open(descend_side_file, 'r+') as r:
                        for line in r.readlines():
                            descend_new_labels_list.append(line.strip())

                label_dict = Label_Vocab(descend_new_labels_list)

                joblib.dump(label_dict, side_label_dict_file)
                print(descend_new_labels_list)
                print(len(descend_new_labels_list))
            else:
                label_dict = joblib.load(side_label_dict_file)
            og_to_new = generate_og_to_new_dict()
            cano_side_smiles_list = [og_to_new.get(i, '<eos>') for i in cano_side_smiles_list]
            labels_list = [label_dict.get(cano_smiles, label_dict.get('<eos>')) for cano_smiles
                           in cano_side_smiles_list]
            df['label'] = labels_list
            df['sideprod'] = cano_side_smiles_list
            for idx, cano_side_smiles in enumerate(cano_side_smiles_list):
                if cano_side_smiles is not '<eos>':
                    new_rxn = rxn_list[idx] + '.' + cano_side_smiles
                else:
                    new_rxn = rxn_list[idx]
                new_rxn_list.append(new_rxn)
            df['rxn_smiles'] = new_rxn_list
            df.to_csv('%s/side_product_rxn_%s.csv' % (data_name, data_name), index=False)
        else:
            print('side_product_rxn_%s.csv already exists !' % data_name)
