#pip install keras rdkit-pypi py3dmol


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import matplotlib.pyplot as plt

def visualize_molecule_3d(molecule_smiles):
    mol = Chem.MolFromSmiles(molecule_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    block = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=800, height=400)
    viewer.addModel(block, "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    return viewer.show()

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return adjacency_matrix, atom_features

def build_mpnn_model(num_atom_features, num_edge_features, message_passing_steps):
    atom_input = layers.Input(shape=(num_atom_features,))
    edge_input = layers.Input(shape=(None, num_edge_features))
    adjacency_input = layers.Input(shape=(None,))

    x = atom_input
    for step in range(message_passing_steps):
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, atom_input])

    output = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=[atom_input, edge_input, adjacency_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


smiles_list = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "CC1=CC=CC=C1"]
graphs = [smiles_to_graph(s) for s in smiles_list]

max_atoms = max([g[1] for g in graphs])
atom_features = np.array([np.pad(g[1], (0, max_atoms - len(g[1]))) for g in graphs])
adjacency_matrices = np.array([np.pad(g[0], ((0, max_atoms - g[0].shape[0]), (0, max_atoms - g[0].shape[1]))) for g in graphs])
edges_dummy = np.zeros((len(smiles_list), max_atoms, max_atoms))

mpnn_model = build_mpnn_model(num_atom_features=max_atoms, num_edge_features=1, message_passing_steps=4)
history = mpnn_model.fit([atom_features, edges_dummy, adjacency_matrices], np.random.rand(len(smiles_list)), epochs=10, verbose=1)

plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Treinamento MPNN para moléculas complexas')
plt.show()

visualize_molecule_3d("CC(=O)OC1=CC=CC=C1C(=O)O")
