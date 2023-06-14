from rdkit import Chem
import networkx as nx
import numpy as np
import torch

class Label_Vocab:
    """Vocab class to deal with atom vocabularies and other attributes."""

    def __init__(self, elem_list) -> None:
        """
        Parameters
        ----------
        elem_list: List, default ATOM_LIST
            Element list used for setting up the vocab
        """
        self.elem_list = elem_list
        if isinstance(elem_list, dict):
            self.elem_list = list(elem_list.keys())
        self.elem_to_idx = {a: idx for idx, a in enumerate(self.elem_list)}
        self.idx_to_elem = {idx: a for idx, a in enumerate(self.elem_list)}

    def __getitem__(self, a_type: str) -> int:
        return self.elem_to_idx[a_type]

    def get(self, elem: str, idx: int = None) -> int:
        """Returns the index of the element, else a None for missing element.

        Parameters
        ----------
        elem: str,
            Element to query
        idx: int, default None
            Index to return if element not in vocab
        """
        return self.elem_to_idx.get(elem, idx)

    def get_elem(self, idx: int) -> str:
        """Returns the element at given index.

        Parameters
        ----------
        idx: int,
            Index to return if element not in vocab
        """
        return self.idx_to_elem[idx]

    def __len__(self) -> int:
        return len(self.elem_list)

    def index(self, elem: str) -> int:
        """Returns the index of the element.

        Parameters
        ----------
        elem: str,
            Element to query
        """
        return self.elem_to_idx[elem]

    def size(self) -> int:
        """Returns length of Vocab."""
        return len(self.elem_list)

class RxnElement:
    """
    RxnElement is an abstract class for dealing with single molecule. The graph
    and corresponding molecule attributes are built for the molecule. The constructor
    accepts only mol objects, sidestepping the use of SMILES string which may always
    not be achievable, especially for a unkekulizable molecule.
    """

    def __init__(self, mol: Chem.Mol) -> None:
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        """

        self.smiles = Chem.MolToSmiles(mol)
        smiles_ls = self.smiles.split('.')
        new_smiles_ls = []
        add = False
        for smiles in smiles_ls:
            mol = Chem.MolFromSmiles(smiles)
            if mol.GetNumAtoms() > 1:
                pass
            else:
                add = True
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() in ['Cl','Br','F','I']:
                        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
                    elif atom.GetSymbol() in ['O']:
                        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 2)
                        atom.SetFormalCharge(atom.GetFormalCharge() + 2)
                    elif atom.GetSymbol() in ['N']:
                        atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 3)
                        atom.SetFormalCharge(atom.GetFormalCharge() + 3)
            new_smiles_ls.append(Chem.MolToSmiles(mol))
        new_smiles = '.'.join(new_smiles_ls)
        if add:
            self.mol = Chem.AddHs(Chem.MolFromSmiles(new_smiles))
        else:
            self.mol = Chem.MolFromSmiles(new_smiles)

        # self.mol = mol
        # self.rxn_class = rxn_class
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
        """Updates the atom indices by the offset.

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
        """Updates the atom indices by the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        if isinstance(self.bond_scope, list):
            return [(st + offset, le) for (st, le) in self.bond_scope]
        st, le = self.bond_scope
        return (st + offset, le)

def onek_encoding_unk(x, allowable_set):
    """Converts x to one hot encoding.

    Parameters
    ----------
    x: Any,
        An element of any type
    allowable_set: Union[List, Set]
        Allowable element collection
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))

def get_atom_features(atom, rxn_class=None, use_rxn_class=False):
    """Get atom features.

    Parameters
    ----------
    atom: Chem.Atom,
        Atom object from RDKit
    rxn_class: int, None
        Reaction class the molecule was part of
    use_rxn_class: bool, default False,
        Whether to use reaction class as additional input
    """
    if atom.GetSymbol() == '*':
        symbol = onek_encoding_unk(atom.GetSymbol(), ATOM_LIST)
        if use_rxn_class:
            padding = [0] * (ATOM_FDIM + len(RXN_CLASSES) - len(symbol))
        else:
            padding = [0] * (ATOM_FDIM - len(symbol))
        feature_array = symbol + padding
        return feature_array

    if use_rxn_class:
        return np.array(onek_encoding_unk(atom.GetSymbol(), ATOM_LIST) +
                        onek_encoding_unk(atom.GetDegree(), DEGREES) +
                        onek_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGE) +
                        onek_encoding_unk(atom.GetHybridization(), HYBRIDIZATION) +
                        onek_encoding_unk(atom.GetTotalValence(), VALENCE) +
                        onek_encoding_unk(atom.GetTotalNumHs(), NUM_Hs) +
                        [float(atom.GetIsAromatic())] + onek_encoding_unk(rxn_class, RXN_CLASSES)).tolist()

    else:
        return np.array(onek_encoding_unk(atom.GetSymbol(), ATOM_LIST) +
                        onek_encoding_unk(atom.GetDegree(), DEGREES) +
                        onek_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGE) +
                        onek_encoding_unk(atom.GetHybridization(), HYBRIDIZATION) +
                        onek_encoding_unk(atom.GetTotalValence(), VALENCE) +
                        onek_encoding_unk(atom.GetTotalNumHs(), NUM_Hs) +
                        [float(atom.GetIsAromatic())]).tolist()


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """Get bond features.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object
    """
    bt = bond.GetBondType()
    bond_features = [float(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    bond_features.extend([float(bond.GetIsConjugated()), float(bond.IsInRing())])
    bond_features = np.array(bond_features, dtype=np.float32)
    return bond_features


def pack_graph_feats(graph_batch, directed, use_rxn_class=False, rxn_class=None):
    """Prepare graph tensors.

    Parameters
    ----------
    graph_batch: List[Any],
        Batch of graph objects. Should have attributes G_dir, G_undir
    directed: bool,
        Whether to prepare tensors for directed message passing
    use_rxn_class: bool, default False,
        Whether to use reaction class as additional input
    return_graphs: bool, default False,
        Whether to return the graphs
    """

    fnode = []
    fmess = []
    agraph, bgraph = [], []

    atom_scope, bond_scope = [], []
    edge_dict = {}
    all_G = []

    for bid, graph in enumerate([graph_batch]):
        mol = graph.mol
        assert mol.GetNumAtoms() == len(graph.G_dir)
        assert len(graph.G_dir) >= 1
        if len(graph.G_dir) == 1:
            sm = Chem.MolToSmiles(mol)
            print(sm)

        atom_offset = len(fnode)
        atom_scope.append(graph.update_atom_scope(atom_offset))

        G = nx.convert_node_labels_to_integers(graph.G_dir, first_label=atom_offset)
        all_G.append(G)
        fnode.extend([None for v in G.nodes])

        for v, attr in G.nodes(data='label'):
            G.nodes[v]['batch_id'] = bid
            fnode[v] = get_atom_features(mol.GetAtomWithIdx(v - atom_offset), use_rxn_class=use_rxn_class,
                                         rxn_class=rxn_class)
            agraph.append([])

        for u, v, attr in G.edges(data='label'):
            bond_feat = get_bond_features(mol.GetBondBetweenAtoms(u - atom_offset, v - atom_offset)).tolist()
            mess_vec = [u + 1, v + 1] + bond_feat

            fmess.append(mess_vec)
            edge_dict[(u, v)] = eid = len(edge_dict)
            G[u][v]['mess_idx'] = eid
            agraph[v].append(eid + 1)
            bgraph.append([])

        for u, v in G.edges:
            eid = edge_dict[(u, v)]
            for w in G.predecessors(u):
                if w == v: continue
                bgraph[eid].append(edge_dict[(w, u)] + 1)
    if len(fmess) == 0:
        fmess.append([1,2,0,0,0,0,0,0])
        bgraph = [[]]
        print('!!!!!!!!!!!!!!!!!')
    fnode = torch.tensor(fnode, dtype=torch.float)
    fmess = torch.tensor(fmess, dtype=torch.float)
    graph_tensors = (fnode, fmess, agraph, bgraph)

    return graph_tensors, atom_scope



BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs', '*', 'unk']
MAX_NB = 10
DEGREES = list(range(MAX_NB))
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]

FORMAL_CHARGE = [-1, -2, 1, 2, 0]
VALENCE = [0, 1, 2, 3, 4, 5, 6]
NUM_Hs = [0, 1, 3, 4, 5]
ATOM_FDIM = len(ATOM_LIST) + len(DEGREES) + len(FORMAL_CHARGE) + len(HYBRIDIZATION) \
            + len(VALENCE) + len(NUM_Hs) + 1
BOND_FDIM = 6

MAX_VALENCE = {'N': 3, 'C': 4, 'O': 2, 'Br': 1, 'Cl': 1, 'F': 1, 'I': 1}
VALENCE_CHECK = {'Na': 1, 'Li': 1, 'K': 1, 'Mg': 2, 'B': 3, 'C': 4}
MAX_VALENCE.update(VALENCE_CHECK)

SINGLE_BOND_ATTACH_GROUPS = ['Br[C:1](Br)Br', 'Br[Zn:1]', '[Cl:1]', '[F:1].[O:1]',
                             '[Cu:1]', '[F:1]', '[I:1]', '[Br:1]', 'O[B:1]O', 'Br[CH:1]Br', 'Br[Mg:1]', 'C1CC[CH:1]OC1',
                             'C1CO[B:1]O1', 'C1CO[B:1]OC1', 'C=C(C)C(=O)[O:1]', 'C=C1CNC([O:1])O1',
                             'C=CB1OB(C=C)O[B:1]O1',
                             'C=CCO[C:1]=O', 'CB1OB(C)O[B:1]O1', 'CC(=O)C(C)[O:1]', 'CC(=O)O[C@@H](c1ccccc1)[C:1]=O',
                             'CC(=O)[O:1]', 'CC(C)(C)C(=O)[O:1]', 'CC(C)(C)OC(=O)[O:1]', 'CC(C)(C)O[C:1]=O',
                             'CC(C)(C)[C:1]=O',
                             'CC(C)(C)[O:1]', 'CC(C)(C)[Si:1](C)C', 'CC(C)(C[O:1])NS(=O)(=O)C(F)(F)F',
                             'CC(C)C(=O)[O:1]',
                             'CC(C)C(C)(C)[Si:1](C)C', 'CC(C)C[O:1]', 'CC(C)O[B:1]OC(C)C', 'CC(C)[C:1]=O', 'CC(C)[O:1]',
                             'CC(C)[Si:1](C(C)C)C(C)C', 'CC(Cl)[O:1]', 'CC1(C)CO[B:1]OC1', 'CC1(C)O[B:1]OC1(C)C',
                             'CC1([C:1]=O)CCCCC1', 'CC1CC(C)(C)O[B:1]O1', 'CC1O[B:1]OC1C', 'CCC(=O)[O:1]',
                             'CCCC(=O)[O:1]',
                             'CCCCC(=O)[O:1]', 'CCCCCC(=O)[O:1]', 'CCCCCCCCC(=O)[O:1]',
                             'CCCCCCCCCCCCCCCCCCCCC(=O)[O:1]',
                             'CCCCCCCCCCCCCCC[C:1]=O', 'CCCCCCCCCCCC[O:1]', 'CCCCCCCOc1ccc([C:1]=O)cc1', 'CCCC[C:1]=O',
                             'CCCC[C:1]Cl', 'CCCC[O:1]', 'CCCC[Sn:1](CCCC)CCCC', 'CCC[C:1]=O', 'CCC[O:1]',
                             'CCOC(=O)[O:1]',
                             'CCO[C:1]=O', 'CCO[P:1](=O)OCC', 'CC[C:1]=O', 'CC[O:1]', 'CC[Si:1](CC)CC',
                             'CC[Sn:1](CC)CC',
                             'CN(C)[C:1]=O', 'CN1CC(=O)O[B:1]OC(=O)C1', 'COC(=O)CC[C:1]=O', 'CO[C:1]=O', 'CO[P:1](=O)OC'
                                                                                                         'CON=C(C(=O)[O:1])c1csc(NC(c2ccccc2)(c2ccccc2)c2ccccc2)n1',
                             'CO[CH2:1]', 'CO[N:1]C',
                             'COc1cc(-c2ncn(C=CC(=O)[O:1])n2)cc(C(F)(F)F)c1', 'COc1ccc([C:1]=O)cc1',
                             'COc1ccc([CH2:1])cc1',
                             'COc1ccc2cc([C@@H](C)[C:1]=O)ccc2c1', 'CS(=O)(=O)[O:1]', 'C[C:1](C)C', 'C[C:1]=O',
                             'C[CH2:1]',
                             'C[CH:1]C', 'C[N+:1](C)C', 'C[N:1]C', 'C[O:1]', 'C[P+:1](C)C', 'C[S+:1](C)=O', 'C[S+:1]C',
                             'C[S:1](=O)=O', 'C[Si:1](C)C', 'C[Si](C)(C)[N:1]', 'C[Sn:1](C)C', 'ClC(Cl)(Cl)C[O:1]',
                             'ClC(Cl)(Cl)[O:1]', 'Cl[C:1](Cl)Cl', 'FC(F)(F)C[O:1]',
                             'Fc1ccc(B2OB(c3ccc(F)cc3)O[B:1]O2)cc1',
                             'I[CH2:1]', 'I[Zn:1]', 'O=C(C(F)(F)Cl)[O:1]', 'O=C(C(F)(F)F)[O:1]', 'O=C(C(F)F)[O:1]',
                             'O=C(C1CC1)[O:1]', 'O=C(CBr)[O:1]', 'O=C(CCl)[O:1]', 'O=C(CI)[O:1]',
                             'O=C(O)C=CC=CC=CC=C[C:1]=O',
                             'O=C(OCc1ccccc1)[O:1]', 'O=C([O-])C1C=CO[B:1]O1', 'O=C(c1cccc(Cl)c1)[O:1]',
                             'O=C(c1ccccc1)[O:1]',
                             'O=C(c1ccccc1Cl)[O:1]', 'O=C1CCC(=O)[N:1]1', 'O=C[O:1]', 'O=S(=O)(C(F)(F)F)[N:1]',
                             'O=S(=O)(C(F)(F)F)[O:1]', 'O=S([O-])[O:1]', 'O=[C:1]C(F)(F)F', 'O=[C:1]CCCCBr',
                             'O=[C:1]CCl',
                             'O=[C:1]OCC1c2ccccc2-c2ccccc21', 'O=[C:1]OCc1ccccc1', 'O=[C:1]c1cccc(Cl)c1',
                             'O=[C:1]c1ccccc1',
                             'O=[N+]([O-])c1ccc(C[O:1])cc1', 'O=[N+]([O-])c1ccc(O)c([CH2:1])c1',
                             'O=[N+]([O-])c1ccc([C:1]=O)cc1',
                             'O=[P:1](OCC(F)(F)F)OCC(F)(F)F', '[CH3:1]', '[Mg+:1]', '[NH2:1]', '[O-:1]', '[OH:1]',
                             '[Zn+:1]',
                             'c1ccc(C[O:1])cc1', 'c1ccc(N2CCO[B:1]OCC2)cc1', 'c1ccc([CH2:1])cc1',
                             'c1ccc([CH:1]c2ccccc2)cc1',
                             'c1ccc([P+:1](c2ccccc2)c2ccccc2)cc1', "COC[C:1]=O", "O=[C:1]C(F)(F)C(F)(F)C(F)(F)F",
                             "C1CC[N:1]CC1",
                             "CC(C)(C)[Si](C)(C)Oc1ccc(B2OB(c3ccc(O[Si](C)(C)C(C)(C)C)cc3)O[B:1]O2)cc1",
                             "CO[P:1](=O)OC"]

DOUBLE_BOND_ATTACH_GROUPS = ['[N-]=[N+:1]', '[CH2:1]', '[O:1]', '[S:1]',
                             'c1ccc([P:1](c2ccccc2)c2ccccc2)cc1']
SINGLE_ATTACH = SINGLE_BOND_ATTACH_GROUPS + DOUBLE_BOND_ATTACH_GROUPS

BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],}
RXN_CLASSES = list(range(1, 11))

BOND_TYPE = [0,
             Chem.rdchem.BondType.SINGLE,
             Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.QUADRUPLE]




