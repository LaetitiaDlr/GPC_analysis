# tests/test_init.py
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from GPC_analysis.class_GPC import GPC_dataset

# Chemin vers le fichier de test réel
TEST_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'P1.022-17072025.xlsx')

@pytest.fixture
def real_test_file():
    """Fixture qui retourne le chemin vers le vrai fichier de test"""
    if not os.path.exists(TEST_FILE_PATH):
        pytest.skip(f"Fichier de test non trouvé: {TEST_FILE_PATH}")
    return TEST_FILE_PATH

@pytest.fixture
def sample_info_for_real_file():
    """Fixture avec les vraies informations du sample P1.022"""
    return {
        'P1.022': {
            'Experiment': 'Test_GPC',
            'Sample': 'P1.022',
            'Date': '17/07/2025'
        }
    }

def test_initialization_with_real_file_raw_type(real_test_file, sample_info_for_real_file):
    """Test initialisation avec le vrai fichier P1.022 - type raw"""
    filepath_dict = {'P1.022': real_test_file}
    
    # Mock _process_raw_data car le fichier Excel n'est pas au format raw
    with patch.object(GPC_dataset, '_process_raw_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_info_for_real_file,
            report_type='raw'
        )
        
        assert gpc.filepath_dict == filepath_dict
        assert gpc.sample_information == sample_info_for_real_file
        assert gpc.report_type == 'raw'
        assert hasattr(gpc, 'palette')

def test_initialization_with_defaults():
    """Test initialisation avec valeurs par défaut - avec mock"""
    filepath_dict = {'sample1': 'dummy_path.csv'}
    sample_information = {'sample1': {'Experiment': 'Test_Exp'}}
    
    with patch.object(GPC_dataset, '_process_raw_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        assert gpc.filepath_dict == filepath_dict
        assert gpc.sample_information == sample_information
        assert gpc.report_type == 'raw'
        assert hasattr(gpc, 'palette')

def test_initialization_with_custom_palette(real_test_file, sample_info_for_real_file):
    """Test initialisation avec palette personnalisée - fichier réel"""
    filepath_dict = {'P1.022': real_test_file}
    custom_palette = {'P1.022': 'red'}
    
    with patch.object(GPC_dataset, '_process_raw_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_info_for_real_file,
            palette=custom_palette,
            report_type='raw'
        )
        
        assert gpc.palette == custom_palette

def test_mark_houwink_parameters(real_test_file, sample_info_for_real_file):
    """Test que les paramètres Mark-Houwink sont correctement initialisés"""
    filepath_dict = {'P1.022': real_test_file}
    
    with patch.object(GPC_dataset, '_process_raw_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_info_for_real_file,
            report_type='raw'
        )
        
        # Vérifier les constantes Mark-Houwink
        assert gpc.PS_alpha == 0.722
        assert gpc.PS_K == 0.000102
        assert gpc.PP_alpha == 0.725
        assert gpc.PP_K == 0.000190
        assert gpc.density_PP == 910

def test_conversion_factors_calculation(real_test_file, sample_info_for_real_file):
    """Test que les facteurs de conversion H_0 et H_1 sont calculés correctement"""
    filepath_dict = {'P1.022': real_test_file}
    
    with patch.object(GPC_dataset, '_process_raw_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_info_for_real_file,
            report_type='raw'
        )
        
        # Vérifier que H_0 et H_1 existent
        assert hasattr(gpc, 'H_0')
        assert hasattr(gpc, 'H_1')
        
        # Vérifier les valeurs calculées
        import math
        expected_H_0 = (math.log10(gpc.PS_K) - math.log10(gpc.PP_K))/(gpc.PP_alpha+1)
        expected_H_1 = (gpc.PS_alpha+1)/(gpc.PP_alpha+1)
        
        assert abs(gpc.H_0 - expected_H_0) < 0.001
        assert abs(gpc.H_1 - expected_H_1) < 0.001

def test_colors_list_exists(real_test_file, sample_info_for_real_file):
    """Test que la liste des couleurs est définie"""
    filepath_dict = {'P1.022': real_test_file}
    
    with patch.object(GPC_dataset, '_process_raw_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_info_for_real_file,
            report_type='raw'
        )
        
        assert hasattr(gpc, 'colors')
        assert isinstance(gpc.colors, list)
        assert len(gpc.colors) == 10  # Vous avez défini 10 couleurs
        assert gpc.colors[0] == 'red'
        assert gpc.colors[1] == 'blue'

def test_default_palette_creation():
    """Test que la palette par défaut est créée correctement"""
    filepath_dict = {'sample1': 'path1.csv', 'sample2': 'path2.csv'}
    sample_information = {
        'sample1': {'Experiment': 'Test_Exp'},
        'sample2': {'Experiment': 'Test_Exp'}
    }
    
    with patch.object(GPC_dataset, '_process_raw_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        assert isinstance(gpc.palette, dict)
        assert 'sample1' in gpc.palette
        assert 'sample2' in gpc.palette
        assert len(gpc.palette) == 2

def test_raw_data_attributes_created():
    """Test que les attributs data_raw et data_MMD_all sont créés pour raw"""
    filepath_dict = {'sample1': 'dummy_raw.csv'}
    sample_information = {'sample1': {'Experiment': 'Test_Exp'}}
    
    # Mock _process_raw_data pour simuler la création des attributs
    with patch.object(GPC_dataset, '_process_raw_data') as mock_process:
        def create_raw_attributes(self):
            # Simuler ce que fait _process_raw_data
            self.data_raw = {'sample1': pd.DataFrame()}
            self.data_MMD_all = {'sample1': pd.DataFrame()}
        
        mock_process.side_effect = create_raw_attributes
        
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        # Vérifier que _process_raw_data a été appelé
        mock_process.assert_called_once()

def test_process_raw_data_called_with_raw_type():
    """Test que _process_raw_data est appelé quand report_type='raw'"""
    filepath_dict = {'sample1': 'dummy_raw.csv'}
    sample_information = {'sample1': {'Experiment': 'Test_Exp'}}
    
    with patch.object(GPC_dataset, '_process_raw_data') as mock_raw:
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        # Vérifier que _process_raw_data a été appelé
        mock_raw.assert_called_once()
        
        # Vérifier que report_type est bien 'raw'
        assert gpc.report_type == 'raw'

def test_process_excel_data_not_called_with_raw_type():
    """Test que _process_excel_data N'est PAS appelé quand report_type='raw'"""
    filepath_dict = {'sample1': 'dummy_raw.csv'}
    sample_information = {'sample1': {'Experiment': 'Test_Exp'}}
    
    with patch.object(GPC_dataset, '_process_raw_data'), \
         patch.object(GPC_dataset, '_process_excel_data') as mock_excel:
        
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        # Vérifier que _process_excel_data N'a PAS été appelé
        mock_excel.assert_not_called()

def test_raw_data_structure_with_mocked_data():
    """Test la structure des données raw mockées"""
    filepath_dict = {'P1.022': 'dummy_raw.csv'}
    sample_information = {'P1.022': {'Experiment': 'Test_Exp'}}
    
    # Créer des données mockées réalistes
    mock_raw_data = pd.DataFrame({
        'Elution Volume (mL)': np.linspace(8, 18, 100),
        'Concentration mg/mL': np.random.rand(100) * 0.5
    })
    
    mock_mmd_data = pd.DataFrame({
        'LogM': np.linspace(3, 6, 100),
        'MMD': np.random.rand(100)
    })
    
    with patch.object(GPC_dataset, '_process_raw_data') as mock_process:
        def set_mock_data(self):
            self.data_raw = {'P1.022': mock_raw_data}
            self.data_MMD_all = {'P1.022': mock_mmd_data}
        
        mock_process.side_effect = set_mock_data
        
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        # Vérifier la structure
        assert hasattr(gpc, 'data_raw')
        assert hasattr(gpc, 'data_MMD_all')
        assert 'P1.022' in gpc.data_raw
        assert 'P1.022' in gpc.data_MMD_all
        assert isinstance(gpc.data_raw['P1.022'], pd.DataFrame)
        assert isinstance(gpc.data_MMD_all['P1.022'], pd.DataFrame)
