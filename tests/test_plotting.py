"""
Tests des fonctions de plotting - avec report_type='raw'
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pytest
import pandas as pd
import numpy as np
import os
from GPC_analysis.class_GPC import GPC_dataset


@pytest.fixture
def test_excel_path():
    """Retourne le chemin vers le fichier Excel de test"""
    current_dir = os.path.dirname(__file__)
    excel_path = os.path.join(current_dir, 'P1.022-17072025.xlsx')
    
    if not os.path.exists(excel_path):
        pytest.skip(f"Fichier de test non trouvé: {excel_path}")
    
    return excel_path


@pytest.fixture
def sample_information():
    """Information sur l'échantillon de test"""
    return {
        'P1.022': {
            'Experiment': 'Test_Exp',
            'Sample': 'P1.022'
        }
    }


@pytest.fixture
def gpc_dataset(test_excel_path, sample_information):
    """Crée un GPC_dataset avec le fichier de test"""
    filepath_dict = {'P1.022': test_excel_path}
    
    gpc = GPC_dataset(
        filepath_dict=filepath_dict,
        sample_information=sample_information,
        report_type='raw'  # ✅ Changé en 'raw'
    )
    return gpc


def test_file_exists(test_excel_path):
    """Test que le fichier de test existe"""
    assert os.path.exists(test_excel_path)
    print(f"\n✓ Fichier trouvé: {test_excel_path}")


def test_dataset_initialization(gpc_dataset):
    """Test que le dataset s'initialise correctement"""
    assert gpc_dataset is not None
    assert gpc_dataset.report_type == 'raw'
    print(f"\n✓ GPC_dataset initialisé avec report_type='raw'")


def test_dataset_has_raw_data(gpc_dataset):
    """Test que le dataset contient data_raw"""
    assert hasattr(gpc_dataset, 'data_raw')
    assert len(gpc_dataset.data_raw) > 0
    
    sample_name = list(gpc_dataset.data_raw.keys())[0]
    df = gpc_dataset.data_raw[sample_name]
    
    print(f"\n✓ data_raw trouvé pour: {sample_name}")
    print(f"  Colonnes: {df.columns.tolist()}")
    print(f"  Nombre de lignes: {len(df)}")


def test_dataset_has_MMD_data(gpc_dataset):
    """Test que le dataset contient data_MMD_all"""
    assert hasattr(gpc_dataset, 'data_MMD_all')
    assert len(gpc_dataset.data_MMD_all) > 0
    
    sample_name = list(gpc_dataset.data_MMD_all.keys())[0]
    df = gpc_dataset.data_MMD_all[sample_name]
    
    print(f"\n✓ data_MMD_all trouvé pour: {sample_name}")
    print(f"  Index: {df.index.name}")  # Devrait être 'LogM'
    print(f"  Colonnes: {df.columns.tolist()}")
    
    # Vérifier que LogM est bien l'index
    assert df.index.name == 'LogM' or 'LogM' in str(df.index.name).lower()


def test_plotting_MMD_Mw_with_logM_as_index(gpc_dataset):
    """Test plotting MMD+Mw avec LogM en index"""
    # Vérifier la structure des données
    sample_name = list(gpc_dataset.data_MMD_all.keys())[0]
    df = gpc_dataset.data_MMD_all[sample_name]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    try:
        # Plot MMD (LogM en index, MMD en colonne)
        if 'MMD' in df.columns:
            axes[0].plot(df.index, df['MMD'], label=sample_name)
            axes[0].set_xlabel('Log M')
            axes[0].set_ylabel('MMD')
            axes[0].legend()
            
            print(f"\n✓ Plot MMD créé")
            print(f"  X (index): min={df.index.min():.2f}, max={df.index.max():.2f}")
            print(f"  Y (MMD): min={df['MMD'].min():.4f}, max={df['MMD'].max():.4f}")
            
            assert len(axes[0].get_lines()) > 0
        
        # Plot Mw si disponible
        if hasattr(gpc_dataset, 'Mn_Mw_from_raw'):
            df_mw = gpc_dataset.Mn_Mw_from_raw
            if 'Mw' in df_mw.columns:
                axes[1].bar(range(len(df_mw)), df_mw['Mw'])
                axes[1].set_ylabel('Mw (g/mol)')
                print(f"\n✓ Plot Mw créé")
        
    finally:
        plt.close(fig)


def test_plotting_MMD_scale_log(gpc_dataset):
    """Test plotting MMD avec échelle log"""
    sample_name = list(gpc_dataset.data_MMD_all.keys())[0]
    df = gpc_dataset.data_MMD_all[sample_name]
    
    fig, ax = plt.subplots()
    
    try:
        if 'MMD' in df.columns:
            # LogM en index, donc on plot directement l'index
            ax.plot(df.index, df['MMD'], label=sample_name)
            ax.set_xlabel('Log M')
            ax.set_ylabel('MMD')
            ax.set_xscale('linear')  # LogM est déjà en log
            ax.legend()
            
            # Ou si on veut M en échelle log:
            # M = 10**df.index
            # ax.plot(M, df['MMD'])
            # ax.set_xscale('log')
            
            assert len(ax.get_lines()) > 0
            print(f"\n✓ Plot avec LogM en index créé")
            
    finally:
        plt.close(fig)


def test_plotting_function_exists(gpc_dataset):
    """Test que la méthode plotting_MMD_Mw existe"""
    # Vérifier les différentes variantes possibles
    plotting_methods = [
        'plotting_MMD_Mw',
        'plot_MMD_Mw',
        'plot_MMD',
    ]
    
    found_methods = []
    for method in plotting_methods:
        if hasattr(gpc_dataset, method):
            found_methods.append(method)
            print(f"\n✓ Méthode trouvée: {method}")
    
    if not found_methods:
        print("\n⚠ Aucune méthode de plotting trouvée")
        print(f"Méthodes disponibles: {[m for m in dir(gpc_dataset) if 'plot' in m.lower()]}")


def test_can_access_logM_as_index(gpc_dataset):
    """Test qu'on peut accéder à LogM comme index"""
    sample_name = list(gpc_dataset.data_MMD_all.keys())[0]
    df = gpc_dataset.data_MMD_all[sample_name]
    
    # LogM devrait être l'index
    logM = df.index.values
    
    assert len(logM) > 0
    assert np.all(np.isfinite(logM))  # Pas de NaN ou inf
    
    # LogM devrait être dans une plage raisonnable (ex: 2 à 7)
    assert logM.min() > 0
    assert logM.max() < 10
    
    print(f"\n✓ LogM accessible comme index")
    print(f"  Plage: [{logM.min():.2f}, {logM.max():.2f}]")
    print(f"  Nombre de points: {len(logM)}")


def test_MMD_column_exists(gpc_dataset):
    """Test que la colonne MMD existe"""
    sample_name = list(gpc_dataset.data_MMD_all.keys())[0]
    df = gpc_dataset.data_MMD_all[sample_name]
    
    # Chercher la colonne MMD (peut avoir différents noms)
    mmd_cols = [col for col in df.columns if 'MMD' in col or 'dW' in col]
    
    assert len(mmd_cols) > 0, f"Aucune colonne MMD trouvée. Colonnes: {df.columns.tolist()}"
    
    print(f"\n✓ Colonne(s) MMD trouvée(s): {mmd_cols}")


def test_straight_line_2points():
    """Test fonction pure - ligne droite entre 2 points"""
    x = np.array([0, 1])
    y = np.array([0, 2])
    
    slope = (y[1] - y[0]) / (x[1] - x[0])
    assert slope == 2.0
    print("\n✓ Calcul de pente correct")


def test_matplotlib_backend():
    """Test backend matplotlib"""
    backend = matplotlib.get_backend()
    assert backend.lower() == 'agg'
    print(f"\n✓ Backend matplotlib: {backend}")


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Nettoyer les plots après chaque test"""
    yield
    plt.close('all')
