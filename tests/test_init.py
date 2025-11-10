# tests/test_init.py
import os
import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
from GPC_analysis.class_GPC import GPC_dataset

# Chemin vers le fichier de test réel (dans le même dossier que ce fichier de test)
TEST_FILE_PATH = os.path.join(os.path.dirname(__file__), 'P1.022-17072025.xlsx')

@pytest.fixture
def real_test_file():
    """Fixture qui retourne le chemin vers le vrai fichier de test"""
    if not os.path.exists(TEST_FILE_PATH):
        pytest.skip(f"Fichier de test non trouvé: {TEST_FILE_PATH}")
    return TEST_FILE_PATH

@pytest.fixture
def sample_info_P1022():
    """Fixture avec les vraies informations du sample P1.022"""
    return {
        'P1.022': {
            'Experiment': 'Test_GPC',
            'Sample': 'P1.022',
            'Date': '17/07/2025'
        }
    }

# ========================================
# Tests AVEC le vrai fichier (sans mock)
# ========================================

def test_real_initialization_raw_type(real_test_file, sample_info_P1022):
    """Test initialisation RÉELLE avec fichier P1.022 - type raw"""
    filepath_dict = {'P1.022': real_test_file}
    
    # ⚠️ Si ton fichier Excel n'est pas compatible raw, change en 'excel'
    gpc = GPC_dataset(
        filepath_dict=filepath_dict,
        sample_information=sample_info_P1022,
        report_type='raw'  # ← Change ici selon ton fichier
    )
    
    assert gpc.filepath_dict == filepath_dict
    assert gpc.report_type == 'raw'
    assert hasattr(gpc, 'data_MMD_all')
    assert 'P1.022' in gpc.data_MMD_all

def test_real_mark_houwink_parameters(real_test_file, sample_info_P1022):
    """Test paramètres Mark-Houwink avec fichier réel"""
    filepath_dict = {'P1.022': real_test_file}
    
    gpc = GPC_dataset(
        filepath_dict=filepath_dict,
        sample_information=sample_info_P1022,
        report_type='raw'
    )
    
    assert gpc.PS_alpha == 0.722
    assert gpc.PS_K == 0.000102
    assert gpc.PP_alpha == 0.725
    assert gpc.PP_K == 0.000190

def test_real_conversion_factors(real_test_file, sample_info_P1022):
    """Test facteurs de conversion H_0 et H_1 avec fichier réel"""
    filepath_dict = {'P1.022': real_test_file}
    
    gpc = GPC_dataset(
        filepath_dict=filepath_dict,
        sample_information=sample_info_P1022,
        report_type='raw'
    )
    
    import math
    expected_H_0 = (math.log10(gpc.PS_K) - math.log10(gpc.PP_K))/(gpc.PP_alpha+1)
    expected_H_1 = (gpc.PS_alpha+1)/(gpc.PP_alpha+1)
    
    assert abs(gpc.H_0 - expected_H_0) < 0.001
    assert abs(gpc.H_1 - expected_H_1) < 0.001



def test_real_colors_list(real_test_file, sample_info_P1022):
    """Test liste des couleurs avec fichier réel"""
    filepath_dict = {'P1.022': real_test_file}
    
    gpc = GPC_dataset(
        filepath_dict=filepath_dict,
        sample_information=sample_info_P1022,
        report_type='raw'
    )
    
    assert hasattr(gpc, 'colors')
    assert isinstance(gpc.colors, list)
    assert len(gpc.colors) == 10
    assert gpc.colors[0] == 'red'

def test_real_data_structure(real_test_file, sample_info_P1022):
    """Test structure des données chargées depuis fichier réel"""
    filepath_dict = {'P1.022': real_test_file}
    
    gpc = GPC_dataset(
        filepath_dict=filepath_dict,
        sample_information=sample_info_P1022,
        report_type='raw'
    )
    
    # Vérifier data_MMD_all
    assert hasattr(gpc, 'data_MMD_all')
    assert 'P1.022' in gpc.data_MMD_all
    assert isinstance(gpc.data_MMD_all['P1.022'], pd.DataFrame)
    assert not gpc.data_MMD_all['P1.022'].empty
    
    # Vérifier data_Mn_Mw
    assert hasattr(gpc, 'data_Mn_Mw')
    assert isinstance(gpc.data_Mn_Mw, pd.DataFrame)

# ========================================
# Tests AVEC mocks (pour tests unitaires rapides)
# ========================================

def test_initialization_with_mock():
    """Test initialisation avec mock (rapide)"""
    filepath_dict = {'sample1': 'dummy.csv'}
    sample_info = {'sample1': {'Experiment': 'Test'}}
    
    with patch.object(GPC_dataset, '_process_raw_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_info,
            report_type='raw'
        )
        
        assert gpc.report_type == 'raw'

def test_process_raw_called_with_mock():
    """Test que _process_raw_data est appelé"""
    filepath_dict = {'sample1': 'dummy.csv'}
    sample_info = {'sample1': {'Experiment': 'Test'}}
    
    with patch.object(GPC_dataset, '_process_raw_data') as mock_raw:
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_info,
            report_type='raw'
        )
        
        mock_raw.assert_called_once()
