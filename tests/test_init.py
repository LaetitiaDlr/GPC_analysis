# # tests/test_init.py
# import os
# import pytest
# from unittest.mock import patch, MagicMock
# from GPC_analysis.class_GPC import GPC_dataset  # adapte le nom du fichier

# # def test_initialization_with_defaults():
# #     dataset = GPC_dataset(
# #         filepath_dict={'sample1': 'path1.xlsx'},
# #         sample_information={'sample1': {'Experiment': 'E1'}}
# #     )
# #     assert isinstance(dataset, GPC_dataset)
# #     assert dataset.PS_alpha == 0.722
# #     # assert dataset.report_type == 'raw' or not hasattr(dataset, 'report_type')  # pas d'erreur
# filepath_file = os.getcwd()
# test_excel_file = os.path.join(filepath_file, 'P1.022-17072025.xlsx')

# def test_initialization_with_defaults(test_excel_file, sample_information):
#     """Test initialisation avec valeurs par défaut - utilise un vrai fichier"""
#     filepath_dict = {'sample1': test_excel_file}
    
#     with patch.object(GPC_dataset, '_process_excel_data'):
#         gpc = GPC_dataset(
#             filepath_dict=filepath_dict,
#             sample_information=sample_information,
#             report_type='raw'
#         )
        
#         assert gpc.filepath_dict == filepath_dict
#         assert gpc.sample_information == sample_information
#         assert gpc.report_type == 'excel'
#         assert hasattr(gpc, 'palette')


# def test_initialization_with_custom_palette(test_excel_file, sample_information):
#     """Test initialisation avec palette personnalisée"""
#     filepath_dict = {'sample1': test_excel_file}
#     custom_palette = {'sample1': 'red'}
    
#     with patch.object(GPC_dataset, '_process_excel_data'):
#         gpc = GPC_dataset(
#             filepath_dict=filepath_dict,
#             sample_information=sample_information,
#             palette=custom_palette,
#             report_type='raw'
#         )
        
#         assert gpc.palette == custom_palette


# def test_initialization_raw_type(test_raw_file, sample_information):
#     """Test initialisation avec type raw"""
#     filepath_dict = {'sample1': test_raw_file}
    
#     with patch.object(GPC_dataset, '_process_raw_data'):
#         gpc = GPC_dataset(
#             filepath_dict=filepath_dict,
#             sample_information=sample_information,
#             report_type='raw'
#         )
        
#         assert gpc.report_type == 'raw'


# def test_mark_houwink_parameters(test_excel_file, sample_information):
#     """Test que les paramètres Mark-Houwink sont correctement initialisés"""
#     filepath_dict = {'sample1': test_excel_file}
    
#     with patch.object(GPC_dataset, '_process_excel_data'):
#         gpc = GPC_dataset(
#             filepath_dict=filepath_dict,
#             sample_information=sample_information,
#             report_type='raw'
#         )
        
#         assert gpc.PS_alpha == 0.722
#         assert gpc.PS_K == 0.000102
#         assert gpc.PP_alpha == 0.725
#         assert gpc.PP_K == 0.000190
#         assert gpc.density_PP == 910

# tests/test_init.py
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
from GPC_analysis.class_GPC import GPC_dataset


def test_initialization_with_defaults():
    """Test initialisation avec valeurs par défaut - sans fichier réel"""
    filepath_dict = {'sample1': 'dummy_path.xlsx'}
    sample_information = {'sample1': {'Experiment': 'Test_Exp'}}
    
    # Mock la méthode qui lit le fichier
    with patch.object(GPC_dataset, '_process_excel_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        assert gpc.filepath_dict == filepath_dict
        assert gpc.sample_information == sample_information
        assert gpc.report_type == 'excel'
        assert hasattr(gpc, 'palette')


def test_initialization_with_custom_palette():
    """Test initialisation avec palette personnalisée"""
    filepath_dict = {'sample1': 'dummy_path.xlsx'}
    sample_information = {'sample1': {'Experiment': 'Test_Exp'}}
    custom_palette = {'sample1': 'red'}
    
    with patch.object(GPC_dataset, '_process_excel_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            palette=custom_palette,
            report_type='raw'
        )
        
        assert gpc.palette == custom_palette


def test_initialization_raw_type():
    """Test initialisation avec type raw"""
    filepath_dict = {'sample1': 'dummy_path.csv'}
    sample_information = {'sample1': {'Experiment': 'Test_Exp'}}
    
    with patch.object(GPC_dataset, '_process_raw_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        assert gpc.report_type == 'raw'


def test_mark_houwink_parameters():
    """Test que les paramètres Mark-Houwink sont correctement initialisés"""
    filepath_dict = {'sample1': 'dummy_path.xlsx'}
    sample_information = {'sample1': {'Experiment': 'Test_Exp'}}
    
    with patch.object(GPC_dataset, '_process_excel_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        assert gpc.PS_alpha == 0.722
        assert gpc.PS_K == 0.000102
        assert gpc.PP_alpha == 0.725
        assert gpc.PP_K == 0.000190
        assert gpc.density_PP == 910


def test_default_palette_creation():
    """Test que la palette par défaut est créée correctement"""
    filepath_dict = {'sample1': 'path1.xlsx', 'sample2': 'path2.xlsx'}
    sample_information = {
        'sample1': {'Experiment': 'Test_Exp'},
        'sample2': {'Experiment': 'Test_Exp'}
    }
    
    with patch.object(GPC_dataset, '_process_excel_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        assert isinstance(gpc.palette, dict)
        assert 'sample1' in gpc.palette
        assert 'sample2' in gpc.palette
        assert len(gpc.palette) == 2


def test_conversion_factors_calculation():
    """Test que les facteurs de conversion sont calculés correctement"""
    filepath_dict = {'sample1': 'dummy_path.xlsx'}
    sample_information = {'sample1': {'Experiment': 'Test_Exp'}}
    
    with patch.object(GPC_dataset, '_process_excel_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        # Vérifier que H_0 et H_1 existent
        assert hasattr(gpc, 'H_0')
        assert hasattr(gpc, 'H_1')
        
        # Vérifier les valeurs (approximativement)
        import math
        expected_H_0 = (math.log10(gpc.PS_K) - math.log10(gpc.PP_K))/(gpc.PP_alpha+1)
        expected_H_1 = (gpc.PS_alpha+1)/(gpc.PP_alpha+1)
        
        assert abs(gpc.H_0 - expected_H_0) < 0.001
        assert abs(gpc.H_1 - expected_H_1) < 0.001

def test_colors_list_exists():
    """Test que la liste des couleurs est définie"""
    filepath_dict = {'sample1': 'dummy_path.xlsx'}
    sample_information = {'sample1': {'Experiment': 'Test_Exp'}}
    
    with patch.object(GPC_dataset, '_process_excel_data'):
        gpc = GPC_dataset(
            filepath_dict=filepath_dict,
            sample_information=sample_information,
            report_type='raw'
        )
        
        assert hasattr(gpc, 'colors')
        assert isinstance(gpc.colors, list)
        assert len(gpc.colors) > 0
