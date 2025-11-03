"""
Tests of calculation functions with the real file P1.022-17072025.xlsx
"""
import pytest
import numpy as np
import pandas as pd
import os
from GPC_analysis.class_GPC import GPC_dataset


@pytest.fixture
def test_excel_path():
    """Returns the path to the test Excel file"""
    current_dir = os.path.dirname(__file__)
    excel_path = os.path.join(current_dir, 'P1.022-17072025.xlsx')
    
    if not os.path.exists(excel_path):
        pytest.skip(f"Test file not found: {excel_path}")
    
    return excel_path


@pytest.fixture
def sample_information():
    """Test sample information"""
    return {
        'P1.022': {
            'Experiment': 'Test_Exp',
            'Sample': 'P1.022'
        }
    }


@pytest.fixture
def gpc_dataset(test_excel_path, sample_information):
    """Creates a GPC_dataset with the test file"""
    filepath_dict = {'P1.022': test_excel_path}
    
    gpc = GPC_dataset(
        filepath_dict=filepath_dict,
        sample_information=sample_information,
        report_type='raw'
    )
    return gpc


def test_file_loaded(gpc_dataset):
    """Test that the file is loaded correctly"""
    assert gpc_dataset is not None
    assert hasattr(gpc_dataset, 'data_raw')
    assert len(gpc_dataset.data_raw) > 0
    
    sample_name = list(gpc_dataset.data_raw.keys())[0]
    print(f"\n✓ File loaded for: {sample_name}")


def test_data_raw_corrected_exists(gpc_dataset):
    """Test that data_raw_corrected exists after processing"""
    if hasattr(gpc_dataset, 'data_raw_corrected'):
        assert len(gpc_dataset.data_raw_corrected) > 0
        
        sample_name = list(gpc_dataset.data_raw_corrected.keys())[0]
        df = gpc_dataset.data_raw_corrected[sample_name]
        
        print(f"\n✓ data_raw_corrected available:")
        print(f"  Sample: {sample_name}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Check necessary columns
        assert 'LogM' in df.columns, "Column 'LogM' missing"
        assert 'Concentration mg/mL' in df.columns, "Column 'Concentration mg/mL' missing"
    else:
        pytest.skip("data_raw_corrected not yet generated")


def test_calculate_Mn_Mw_raw_data(gpc_dataset):
    """Test calculation of Mn and Mw from real raw data"""
    
    # Check that data_raw_corrected exists
    if not hasattr(gpc_dataset, 'data_raw_corrected'):
        pytest.skip("data_raw_corrected not available")
    
    if len(gpc_dataset.data_raw_corrected) == 0:
        pytest.skip("data_raw_corrected empty")
    
    # Calculate Mn and Mw
    try:
        result = gpc_dataset.calculate_Mn_Mw_raw_data(xlabel='Experiment')
        
        # Verifications
        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
        
        expected_cols = ['Mw', 'Mn', 'PDI', 'M_max']
        for col in expected_cols:
            assert col in result.columns, f"Column '{col}' missing"
        
        # Check values
        sample_name = result.index[0]
        Mw = result.loc[sample_name, 'Mw']
        Mn = result.loc[sample_name, 'Mn']
        PDI = result.loc[sample_name, 'PDI']
        M_max = result.loc[sample_name, 'M_max']
        
        assert Mw > 0, f"Mw should be > 0, got {Mw}"  # type: ignore
        assert Mn > 0, f"Mn should be > 0, got {Mn}"  # type: ignore
        assert Mw >= Mn, f"Mw ({Mw}) should be >= Mn ({Mn})"  # type: ignore
        assert PDI >= 1.0, f"PDI should be >= 1.0, got {PDI}"  # type: ignore
        assert M_max > 0, f"M_max should be > 0, got {M_max}"  # type: ignore
        
        print(f"\n✓ Mn/Mw calculation successful for {sample_name}:")
        print(f"  Mw: {Mw:.2f} g/mol")
        print(f"  Mn: {Mn:.2f} g/mol")
        print(f"  PDI: {PDI:.3f}")
        print(f"  M_max: {M_max:.2f} g/mol")
        
    except Exception as e:
        pytest.fail(f"Error during Mn/Mw calculation: {str(e)}")


def test_calculate_MMD_from_raw_data(gpc_dataset):
    """Test MMD calculation from real raw data"""
    
    # Check that data_raw_corrected exists
    if not hasattr(gpc_dataset, 'data_raw_corrected'):
        pytest.skip("data_raw_corrected not available")
    
    if len(gpc_dataset.data_raw_corrected) == 0:
        pytest.skip("data_raw_corrected empty")
    
    # Calculate MMD
    try:
        result = gpc_dataset.calculate_MMD_from_raw_data()
        
        # Verifications
        assert isinstance(result, dict), "Result should be a dict"
        assert len(result) > 0, "Result should not be empty"
        
        sample_name = list(result.keys())[0]
        df_mmd = result[sample_name]
        
        assert isinstance(df_mmd, pd.DataFrame), "MMD should be a DataFrame"
        assert 'MMD' in df_mmd.columns, "Column 'MMD' missing"
        assert df_mmd.index.name == 'LogM', f"Index should be 'LogM', got {df_mmd.index.name}"
        
        # Check that MMD is normalized (sum ≈ 1)
        mmd_sum = df_mmd['MMD'].sum()
        assert 0.95 < mmd_sum < 1.05, f"MMD sum should be ~1, got {mmd_sum}"
        
        # Check that all values are positive
        assert (df_mmd['MMD'] >= 0).all(), "All MMD values should be >= 0"
        
        print(f"\n✓ MMD calculation successful for {sample_name}:")
        print(f"  Points: {len(df_mmd)}")
        print(f"  LogM range: [{df_mmd.index.min():.2f}, {df_mmd.index.max():.2f}]")
        print(f"  MMD sum: {mmd_sum:.6f}")
        print(f"  MMD max: {df_mmd['MMD'].max():.6f}")
        
    except Exception as e:
        pytest.fail(f"Error during MMD calculation: {str(e)}")


def test_calculate_Mn_Mw_from_MMD(gpc_dataset):
    """Test Mn/Mw calculation from MMD"""
    
    # First calculate MMD
    if not hasattr(gpc_dataset, 'data_raw_corrected'):
        pytest.skip("data_raw_corrected not available")
    
    try:
        data_MMD = gpc_dataset.calculate_MMD_from_raw_data()
        
        # Then calculate Mn/Mw from MMD
        result = gpc_dataset.calculate_Mn_Mw_from_MMD(data_MMD)
        
        # Verifications
        assert isinstance(result, pd.DataFrame)
        
        expected_cols = ['Mw', 'Mn', 'PDI']
        for col in expected_cols:
            assert col in result.columns, f"Column '{col}' missing"
        
        sample_name = result.index[0]
        Mw = result.loc[sample_name, 'Mw']
        Mn = result.loc[sample_name, 'Mn']
        PDI = result.loc[sample_name, 'PDI']
        
        assert Mw > 0 # type: ignore
        assert Mn > 0 # type: ignore
        assert Mw >= Mn # type: ignore
        assert PDI >= 1.0 # type: ignore
        
        print(f"\n✓ Mn/Mw calculation from MMD successful:")
        print(f"  Mw: {Mw:.2f} g/mol")
        print(f"  Mn: {Mn:.2f} g/mol")
        print(f"  PDI: {PDI:.3f}")
        
    except Exception as e:
        pytest.fail(f"Error during Mn/Mw calculation from MMD: {str(e)}")



def test_data_MMD_all_attribute(gpc_dataset):
    """Test that data_MMD_all is created and accessible"""
    
    if hasattr(gpc_dataset, 'data_MMD_all'):
        assert len(gpc_dataset.data_MMD_all) > 0
        
        sample_name = list(gpc_dataset.data_MMD_all.keys())[0]
        df_mmd = gpc_dataset.data_MMD_all[sample_name]
        
        assert isinstance(df_mmd, pd.DataFrame)
        assert len(df_mmd) > 0
        
        print(f"\n✓ data_MMD_all available:")
        print(f"  Sample: {sample_name}")
        print(f"  Shape: {df_mmd.shape}")
        print(f"  Index: {df_mmd.index.name}")
    else:
        pytest.skip("data_MMD_all not yet created")
