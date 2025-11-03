# GPC_analysis

A Python package for GPC (Gel Permeation Chromatography) data analysis and visualization.

## Features

- Process raw GPC data from Excel files
- Automatic baseline correction and noise thresholding
- Calculate molecular weight distributions (MMD)
- Compute Mw, Mn, PDI, and M_max values
- Visualization tools for MMD and molecular weight data
- Mark-Houwink conversion for PS/PP polymers

## Installation

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from GPC_analysis.class_GPC import GPC_dataset

# Define your file paths
filepath_dict = {
    'Sample1': 'path/to/sample1.xlsx',
    'Sample2': 'path/to/sample2.xlsx'
}

# Define sample information (REQUIRED structure)
sample_information = {
    'Sample1': {
        'Experiment': 'Exp1',      # Required field
        'Sample': 'Sample1',       # Optional
        'Milling Time (s)': 3600,  # Optional custom fields
    },
    'Sample2': {
        'Experiment': 'Exp1',
        'Sample': 'Sample2',
    }
}

# Create GPC dataset object
gpc = GPC_dataset(
    filepath_dict=filepath_dict,
    sample_information=sample_information,
    report_type='raw'  # or 'excel' for pre-processed data
)

# Access results
print(gpc.data_Mn_Mw)  # Mw, Mn, PDI, M_max for all samples
print(gpc.data_MMD_all)  # Molecular mass distributions

# Plot results
fig, ax = gpc.plotting_MMD_Mw(gpc.data_MMD_all, scale='log')
plt.show()
```

## Required Input Structure

### 1. filepath_dict

A dictionary mapping sample names to their Excel file paths:

```python
filepath_dict = {
    'SampleName1': 'full/path/to/file1.xlsx',
    'SampleName2': 'full/path/to/file2.xlsx',
}
```

### 2. sample_information (⚠️ CRITICAL)

A dictionary mapping sample names to metadata dictionaries. **Each sample MUST have an 'Experiment' key:**

```python
sample_information = {
    'SampleName1': {
        'Experiment': str,    # REQUIRED - experiment identifier
        'Sample': str,        # Optional - sample identifier  
        # Add any custom fields you need:
        'Milling Time (s)': float,
        'Beads Type': str,
        'Mass of PP (g)': float,
        # ... etc.
    },
    'SampleName2': {
        'Experiment': str,
        'Sample': str,
    }
}
```

**Important Rules:**
- The keys in `filepath_dict` and `sample_information` **MUST match exactly**
- Each sample dict **MUST contain** an `'Experiment'` key
- You can add any custom fields you need for plotting/analysis

### Example with Real Data

```python
from GPC_analysis.class_GPC import GPC_dataset
import matplotlib.pyplot as plt

# Real-world example
filepath_dict = {
    'P1.022': 'data/GPC/P1.022-17072025.xlsx',
    'P1.023': 'data/GPC/P1.023-17072025.xlsx',
    'P1.024': 'data/GPC/P1.024-17072025.xlsx',
}

sample_information = {
    'P1.022': {
        'Experiment': 'DDV418',
        'Sample': 'P1.022',
        'Milling Time (s)': 3600,
        'Beads Type': 'Steel',
        'Mass of PP (g)': 0.5
    },
    'P1.023': {
        'Experiment': 'DDV418',
        'Sample': 'P1.023',
        'Milling Time (s)': 7200,
        'Beads Type': 'Steel',
        'Mass of PP (g)': 0.5
    },
    'P1.024': {
        'Experiment': 'DDV418',
        'Sample': 'P1.024',
        'Milling Time (s)': 10800,
        'Beads Type': 'Steel',
        'Mass of PP (g)': 0.5
    }
}

# Create GPC object
gpc = GPC_dataset(filepath_dict, sample_information, report_type='raw')

# Access calculated results
print("Molecular weights:")
print(gpc.data_Mn_Mw)

# Save results to CSV
gpc.save_results_to_csv('output_folder')

# Plot MMD
fig, ax = gpc.plotting_MMD_Mw(gpc.data_MMD_all, scale='log', xlabel='Milling Time (s)')
plt.show()

# Plot Mw/Mn scatter
fig, ax1, ax2 = gpc.plotting_Mw_Mn_scatter(
    gpc.data_Mn_Mw,
    xlabel='Milling Time (s)',
    label='Beads Type'
)
plt.show()
```

## Error Handling

The package will raise clear errors if:

```python
# ❌ Empty dictionaries
filepath_dict = {}  
# ValueError: filepath_dict cannot be empty. Provide at least one sample file.

# ❌ Mismatched keys
filepath_dict = {'Sample1': 'path1.xlsx'}
sample_information = {'Sample2': {...}}  
# ValueError: Sample names must match between filepath_dict and sample_information.
#   Missing in sample_information: {'Sample1'}
#   Missing in filepath_dict: {'Sample2'}

# ❌ Missing 'Experiment' field
sample_information = {
    'Sample1': {'Sample': 'Sample1'}  # Missing 'Experiment' key
}  
# ValueError: sample_information['Sample1'] must contain 'Experiment' key.
# Expected structure: {'Experiment': str, 'Sample': str, ...}
# Got keys: ['Sample']

# ❌ Wrong types
filepath_dict = ['file1.xlsx']  
# TypeError: filepath_dict must be a dictionary, got list

sample_information = {
    'Sample1': 'not a dict'
}
# TypeError: sample_information['Sample1'] must be a dictionary, got str
```

## Excel File Requirements

Your Excel files must contain a sheet named **'Data'** with these columns:
- `Concentration Smoothed `
- `Retention volume processed mL`
- `Calib NS Volumes mL`
- `Calib LogM Points NS `

(Note: Column names include trailing spaces as exported by GPC software)

## Advanced Usage

### Custom Baseline Correction Parameters

```python
# Access raw data correction with custom parameters
data_corrected = gpc.raw_data_correction(
    gpc.data_converted,
    x_range=[14, 26],          # Data range in minutes
    baseline_window=[10, 31],   # Baseline points in minutes
    correction_plotting_intensity=True  # Show plots
)
```

### Accessing Different Data Stages

```python
# Raw data from Excel
gpc.data_raw

# After elution volume → LogM conversion
gpc.data_converted

# After baseline correction
gpc.data_raw_corrected

# Molecular mass distributions
gpc.data_MMD_all

# Final Mw/Mn/PDI values
gpc.data_Mn_Mw
```

## API Documentation

See the full API documentation at [link to docs]

## License

BSD-3

## Author

Laëtitia Delarue (l.delarue@uu.nl)

## Citation

If you use this package in your research, please cite:
```
[Citation information]
```
