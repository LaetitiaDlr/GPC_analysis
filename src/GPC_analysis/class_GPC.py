import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import numpy as np
# import scipy.differentiate as spi

class GPC_dataset:
    """
    A class to handle GPC dataset analysis and visualization.
    
    This class processes GPC (Gel Permeation Chromatography) data from Excel files,
    performs baseline correction, calculates molecular weight distributions (MMD),
    and computes Mw, Mn, and PDI values.
    
    Parameters
    ----------
    filepath_dict : dict
        Dictionary mapping sample names to their file paths.
        Example: {'Sample1': 'path/to/file1.xlsx', 'Sample2': 'path/to/file2.xlsx'}
        
    sample_information : dict
        Dictionary mapping sample names to their metadata. Each sample must have
        at least an 'Experiment' key. The sample names MUST match the keys in filepath_dict.
        
        **Required structure:**
        {
            'SampleName1': {
                'Experiment': str,  # Required - experiment identifier
                'Sample': str,      # Optional - sample identifier
                # ... other custom fields
            },
            'SampleName2': {
                'Experiment': str,
                'Sample': str,
            }
        }
        
        Example:
        {
            'P1.022': {
                'Experiment': 'DDV418',
                'Sample': 'P1.022',
                'Milling Time (s)': 3600,
                'Beads Type': 'Steel'
            }
        }
        
    palette : dict, optional
        Dictionary mapping sample names to colors for plotting.
        If None, a default color palette will be generated.
        Example: {'Sample1': 'red', 'Sample2': 'blue'}
        
    report_type : str, optional
        Type of report to process. Options are:
        - 'raw' (default): Process raw GPC data with baseline correction
        - 'excel': Use pre-processed data from Excel report
        
    Raises
    ------
    ValueError
        If filepath_dict is empty
        If sample_information is missing required keys
        If sample names don't match between filepath_dict and sample_information
    TypeError
        If inputs are not of the expected types
        
    Examples
    --------
    >>> filepath_dict = {
    ...     'Sample1': 'data/sample1.xlsx',
    ...     'Sample2': 'data/sample2.xlsx'
    ... }
    >>> sample_info = {
    ...     'Sample1': {'Experiment': 'Exp1', 'Sample': 'Sample1'},
    ...     'Sample2': {'Experiment': 'Exp1', 'Sample': 'Sample2'}
    ... }
    >>> gpc = GPC_dataset(filepath_dict, sample_info, report_type='raw')
    """

    def __init__(self, filepath_dict, sample_information,
                palette=None, report_type='raw',
                int_x_range=[14, 26], baseline_window = [10, 31], 
                polymer = 'PP'):
        """
        Initialize GPC dataset analysis.
        
        Parameters
        ----------
        filepath_dict : dict
            Dictionary mapping sample names to their file paths.
            Keys must match those in sample_information.
            
        sample_information : dict
            Dictionary mapping sample names to metadata dictionaries.
            Each sample dict must contain at least 'Experiment' key.
            Keys must match those in filepath_dict.
            
        palette : dict, optional
            Custom color palette for plots. If None, auto-generated.
            
        report_type : str, optional
            'raw' for raw data processing or 'excel' for pre-processed data.
            Default is 'raw'.
            
        Raises
        ------
        ValueError
            If inputs fail validation checks.
        """
        # Validate inputs
        self._validate_inputs(filepath_dict, sample_information, report_type)
        # for integration 
        self.int_x_range = int_x_range
        self.baseline_window = baseline_window
        self.filepath_dict = filepath_dict
        self.sample_information = sample_information
        self.report_type = report_type
        # Mark-Houwink parameters
        self.PS_alpha, self.PS_K = 0.722, 0.000102
        if polymer == 'PP':
            self.PP_alpha, self.PP_K = 0.725, 0.000190
            self.density_PP = 910  # g/L
        
        # Calculate conversion factors
        self.H_0 = (math.log10(self.PS_K) - math.log10(self.PP_K))/(self.PP_alpha+1)
        self.H_1 = (self.PS_alpha+1)/(self.PP_alpha+1)

        # Setup color palette
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 
                    'brown', 'pink', 'gray', 'olive', 'cyan']
        if palette is None:
            self.palette = self._create_default_palette()
        else:
            self.palette = palette
        
        # Process data based on report type


        if report_type == 'excel':
            self._process_excel_data()
        else:  # raw
            self._process_raw_data()

    def _validate_inputs(self, filepath_dict, sample_information, report_type):
        """
        Validate input parameters for GPC_dataset initialization.
        
        Parameters
        ----------
        filepath_dict : dict
            Dictionary mapping sample names to file paths
        sample_information : dict
            Dictionary mapping sample names to metadata dictionaries
        report_type : str
            Type of report ('raw' or 'excel')
            
        Raises
        ------
        TypeError
            If inputs are not of expected types
        ValueError
            If dictionaries are empty, keys don't match, or required fields are missing
        """
        # Check types
        if not isinstance(filepath_dict, dict):
            raise TypeError(f"filepath_dict must be a dictionary, got {type(filepath_dict).__name__}")
        
        if not isinstance(sample_information, dict):
            raise TypeError(f"sample_information must be a dictionary, got {type(sample_information).__name__}")
        
        if not isinstance(report_type, str):
            raise TypeError(f"report_type must be a string, got {type(report_type).__name__}")
        
        # Check that dictionaries are not empty
        if not filepath_dict:
            raise ValueError("filepath_dict cannot be empty. Provide at least one sample file.")
        
        if not sample_information:
            raise ValueError("sample_information cannot be empty. Provide metadata for all samples.")
        
        # Check that keys match between filepath_dict and sample_information
        filepath_keys = set(filepath_dict.keys())
        sample_keys = set(sample_information.keys())
        
        if filepath_keys != sample_keys:
            missing_in_sample_info = filepath_keys - sample_keys
            missing_in_filepath = sample_keys - filepath_keys
            
            error_msg = "Sample names must match between filepath_dict and sample_information.\n"
            if missing_in_sample_info:
                error_msg += f"  Missing in sample_information: {missing_in_sample_info}\n"
            if missing_in_filepath:
                error_msg += f"  Missing in filepath_dict: {missing_in_filepath}\n"
            raise ValueError(error_msg)
        
        # Check that each sample has required 'Experiment' field
        for sample_name, info in sample_information.items():
            if not isinstance(info, dict):
                raise TypeError(
                    f"sample_information['{sample_name}'] must be a dictionary, "
                    f"got {type(info).__name__}"
                )
            
            if 'Experiment' not in info:
                raise ValueError(
                    f"sample_information['{sample_name}'] must contain 'Experiment' key.\n"
                    f"Expected structure: {{'Experiment': str, 'Sample': str, ...}}\n"
                    f"Got keys: {list(info.keys())}"
                )
        
        # Check report_type value
        if report_type not in ['raw', 'excel']:
            raise ValueError(
                f"report_type must be 'raw' or 'excel', got '{report_type}'"
            )

    def _create_default_palette(self):
        """Create default color palette if none provided."""
        palette = {}
        colors = plt.cm.get_cmap('tab10', len(self.filepath_dict))
        for i, file in enumerate(self.filepath_dict.keys()):
            palette[file] = colors(i)
        return palette

    def _process_excel_data(self):
        """Process data from Excel reports."""
        self.data_MMD_all = self.extract_MMD_from_excel()
        self.data_Mn_Mw_software = self.extract_Mw_Mn_from_excel()
        self.data_Mn_Mw = self.calculate_Mn_Mw_from_MMD(self.data_MMD_all)

    def _process_raw_data(self):
        """Process raw data."""
        self.data_raw = self.extract_raw_data(self.filepath_dict)
        self.data_converted = self.convert_ev_to_logM(self.data_raw)
        self.data_raw_corrected = self.raw_data_correction(
            self.data_converted, 
            columns_to_correct=['Concentration mg/mL'],
            correction_plotting_intensity = False
        )
        self.data_MMD_all = self.calculate_MMD_from_raw_data()
        self.data_Mn_Mw = self.calculate_Mn_Mw_raw_data()

    def calculate_Mn_Mw_raw_data(self, xlabel='Experiment', data_raw_corrected=None):
        """Calculate Mw, Mn, PDI, and M_max from corrected raw data.
         Parameters
         ----------
         data_raw_corrected : dict, optional
             Dictionary containing corrected raw data (DataFrame) for each sample. If None, uses self.data_raw_corrected.
         Returns
         -------
         pd.DataFrame
             DataFrame containing Mw, Mn, PDI, and M_max for each sample.
         """
        Mn_Mw_from_raw = {}
        data_raw_corrected = data_raw_corrected.copy() if data_raw_corrected is not None else self.data_raw_corrected.copy()
        for sample_name, df in data_raw_corrected.items():
            intensity = df['Concentration mg/mL']
            logMi = df['LogM']
            Mi = 10 ** logMi  # Convert logM to M
            Mw = sum(Mi*intensity) / sum(intensity) if sum(intensity) > 0 else 0  # Weight-average molar mass
            Mn = sum(intensity) / sum(intensity/Mi) if sum(intensity/Mi) > 0 else 0  # Number-average molar mass
            M_max = intensity.idxmax()  # M at maximum intensity
            PDI = Mw/Mn if Mn > 0 else 0
            info = self.sample_information[sample_name][xlabel]
            Mn_Mw_from_raw[sample_name] = [Mw, Mn, PDI, M_max]
        self.Mn_Mw_from_raw = pd.DataFrame(Mn_Mw_from_raw).T
        self.Mn_Mw_from_raw.columns = ['Mw', 'Mn', 'PDI', 'M_max']
        return self.Mn_Mw_from_raw

    def extract_raw_data(self, filepath_dict):
        """Extract raw data from Excel files. Extracts concentration, retention volume and calibration data.
        Parameters
        ----------
        files : list
            List of file names (reference and samples) to extract data from.
        filepath : dict
            Dictionary mapping file names to their full paths.
        Returns
        -------
        dict
            Dictionary containing raw data (DataFrame) for each sample.
        """
        data_raw_all = {}
    
        for file, path in filepath_dict.items():
            # Check if 'Data' sheet exists
            try:
                excel_file = pd.ExcelFile(path)
                available_sheets = excel_file.sheet_names
                
                if 'Data' not in available_sheets:
                    raise ValueError(
                        f"❌ Sheet 'Data' not found in file: {file}\n"
                        f"   File path: {path}\n"
                        f"   Available sheets: {available_sheets}"
                    )
                
                data_i = pd.read_excel(path, sheet_name='Data', header=0, index_col=0)
                
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"❌ File not found: {file}\n"
                    f"   Path: {path}"
                )
            except Exception as e:
                raise Exception(
                    f"❌ Error reading file: {file}\n"
                    f"   Path: {path}\n"
                    f"   Error: {str(e)}"
                )
            
            required_columns = ['Concentration Smoothed ', 'Retention volume processed mL', 'Calib NS Volumes mL', 'Calib LogM Points NS ']
            missing_columns = [col for col in required_columns if col not in data_i.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in {file}: {missing_columns}")
            df_i = data_i[['Concentration Smoothed ', 'Retention volume processed mL', 'Calib NS Volumes mL', 'Calib LogM Points NS ']]
            df_i.columns = ['Concentration mg/mL', 'Elution Volume (mL)', 'Calib NS Volumes mL', 'Calib LogM Points NS ']
            df_i = df_i.copy()
            data_raw_all[file] = df_i
        return data_raw_all
    
    def convert_ev_to_logM(self, data_raw):
        """Convert elution volume to logM using calibration data.
        Parameters
        ----------
        data_raw : dict
            Dictionary created via extract_raw_data() containing raw data (DataFrame) for each sample.
        Returns
        -------
        dict
            Dictionary containing converted data (DataFrame) for each sample with LogM column added without calibration columns.
        """
        data_converted = {}
        for sample_name, df in data_raw.items():
        
            df = df.copy()
            df.columns = [c.strip() for c in df.columns]

            # Expected columns
            req = ['Calib NS Volumes mL', 'Calib LogM Points NS', 'Elution Volume (mL)']
            missing = [c for c in req if c not in df.columns]
            if missing:
                raise KeyError(f"Missing columns: {missing}. Present columns: {list(df.columns)}")

            x_calib = df['Calib NS Volumes mL']
            x_calib = x_calib.dropna()
            y_calib = df['Calib LogM Points NS']
            y_calib = y_calib.dropna()
            coeffs = np.polyfit(x_calib, y_calib, 3)

            ev = df['Elution Volume (mL)']
            logMPS = coeffs[0]*ev**3 + coeffs[1]*ev**2 + coeffs[2]*ev + coeffs[3]
            logMPP = self.H_0 + self.H_1 * logMPS

            out = df.drop(columns=['Calib NS Volumes mL', 'Calib LogM Points NS']).copy()
            out['LogM'] = logMPP
            data_converted[sample_name] = out
        return data_converted

    def plotting_raw_data(self, data=None, xlabel='Elution Volume (mL)', ylabel='Concentration mg/mL'):
        """Plot raw data for each sample.
        Parameters
        ----------
        data : dict, optional
            Dictionary containing raw data (DataFrame) for each sample. If None, uses self.data_raw.
        xlabel : str, optional
            Label for the x-axis. Default is 'Elution Volume (mL)'. Can be 'LogM' and will set xscale to log.
        ylabel : str, optional
            Label for the y-axis. Default is 'Concentration mg/mL'.
        """
        if data is None:
            data = self.data_raw

        fig,ax = plt.subplots(figsize=(6, 5))
        for sample_name, df in data.items():
            if xlabel not in df.index.names:
                df.index = df[xlabel]
            if xlabel == 'LogM':
                ax.set_xscale('log')
            ax.plot(df.index, df[ylabel], label = sample_name)#, color = self.palette[self.label[sample_name]])
        ax.set_xlabel(f'{xlabel}', fontweight='bold')
        ax.axvline(x=26, color='black', linestyle='--')
        ax.axvline(x=31, color='red', linestyle='--')
        ax.axvline(x=14, color='red', linestyle='--')
        # ax.set_xlim(1,31)
        ax.set_ylabel(r'$\bf{Intensity}\ \it{(mg/mL)}$')
        ax.set_title('Raw data, integration range between red and black lines and baseline calculated between the two red lines')
        plt.legend()

    def straight_line_2points(self, x_range, df, average = True):

        """Generate a straight line between two points defined by x_range in the DataFrame df for baseline correction.
        Parameters
        ----------
        x_range : list or tuple
            A list or tuple containing two x-values [x1, x2] to define the points for the straight line.
        df : pd.DataFrame or pd.Series
            DataFrame or Series containing the data with x-values as index.
        average : bool, optional
            If True, average the y-values around x1 and x2 within a small window (0.5).
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the y-values of the straight line at each x-value in df."""

        if len(x_range) == 0 :#or df.isna().all():
            raise ValueError("x_range or df values is empty or is all NaN.")
        x1, x2 = min(x_range), max(x_range)
        if average:
            range_x1 = df.index[(df.index >= x1 - 0.5) & (df.index <= x1 + 0.5)]
            range_x2 = df.index[(df.index >= x2 - 0.4) & (df.index <= x2 + 0.4)]
            average_x1 = np.mean(range_x1)
            average_x2 = np.mean(range_x2)

            idx1 = (pd.Series(df.index) - average_x1).abs().idxmin() # find index of the closest value to average_x1
            idx2 = (pd.Series(df.index) - average_x2).abs().idxmin() # find index of the closest value to average_x2
        else:
            idx1 = (pd.Series(df.index) - x1).abs().idxmin() # find index of the closest value to x1
            idx2 = (pd.Series(df.index) - x2).abs().idxmin() # find index of the closest value to x2

        # Get the actual x and y values matching these indices
        x1_real = df.index[idx1]
        y1 = df.iloc[idx1]#, 0]
        x2_real = df.index[idx2]
        y2 = df.iloc[idx2]#, 0]

        # Line equation calculation : y = ax + b
        a = (y2 - y1) / (x2_real - x1_real)
        b = y1 - a * x1_real

        # Generate line points
        x_vals = df.index.values
        y_vals = a * x_vals + b

        # Return the line as a DataFrame or Series
        straight_line = pd.DataFrame({'y': y_vals}, index=df.index)
        return straight_line

    def keep_largest_non_nan_block(self, df, col):
        """Keep the largest non-NaN block in a DataFrame column. Used after thresholding to retain the main peak."""
        mask = df[col].notna()

        # Find the changes True/False
        changes = mask.ne(mask.shift())
        block_ids = changes.cumsum()

        # Keep only the True (non-NaN) blocks
        valid_blocks = block_ids[mask]
        
        if len(valid_blocks) == 0:
            return df.iloc[0:0]  # DataFrame empty
        
        # Find the largest block
        largest_block_id = valid_blocks.value_counts().idxmax()
        
        return df[block_ids == largest_block_id]


    def raw_data_correction(self, data_extracted, columns_to_correct = ['MMD', 'Concentration mg/mL'], average_baseline = True, 
                            # threshold_at = 0.003, 
                            replace_by = np.nan, 
                            correction_plotting_MMD = False, correction_plotting_intensity = False):
        """Correct raw data by baseline subtraction and thresholding. The thresholding is done based on the noise level in the baseline region, keeping only the largest non-NaN block after thresholding.
        Parameters
        ----------
        data_extracted : dict
            Dictionary containing extracted data (DataFrame) for each sample.
        columns_to_correct : list
            List of column names to apply the correction on.
        average_baseline : bool, optional
            If True, average the y-values around the baseline points within a small window (0.5).
        x_range : list or tuple, optional
            Two-element list or tuple containing the x-axis limits for the main data range.
        baseline_window : list or tuple, optional
            Two-element list or tuple containing the x-axis limits for the baseline points.
        replace_by : float or np.nan, optional
            Value to replace data points below the threshold. Default is np.nan.
        correction_plotting_MMD : bool, optional
            If True, plot the MMD correction, with the raw, corrected curves and the baseline.
        correction_plotting_intensity : bool, optional
            If True, plot the intensity correction process, with the raw, corrected curves and the baseline.
        Returns
        -------
        dict
            Dictionary containing corrected data (DataFrame) for each sample."""
        data_corrected = {}
        axbaseline_MMD = None
        axbaseline_intensity = None
        x_range = self.int_x_range
        baseline_window = self.baseline_window
        
        if correction_plotting_MMD:
            figbaseline_MMD, axbaseline_MMD = plt.subplots(figsize=(6, 5))
        if correction_plotting_intensity:
            figbaseline_intensity, axbaseline_intensity = plt.subplots(figsize=(6, 5))

        for sample_name, df in data_extracted.items():
            df_i = df.copy()
            df_i = df_i.set_index('Elution Volume (mL)')

            # df_i = df_i[(df_i.index >= min(x_range)) & (df_i.index <= max(x_range))]
            baseline = None  # Initialize baseline outside the loop
            threshold = None  # Initialize threshold outside the loop
            
            for col in columns_to_correct:
            # making the baseline correction
                baseline = self.straight_line_2points(baseline_window, df_i[col], average=average_baseline)
                # baseline = baseline[(baseline.index >= min(x_range)) & (baseline.index <= max(x_range))]

                df_i[col] = df_i[col] - baseline['y']   # Correcting the data by subtracting the baseline
                df_i.loc[df_i[col] < 0, col] = 0 # set negative values to 0
                df_noise = df_i[(df_i.index >= min(baseline_window)) & (df_i.index <= min(x_range))] #looking for the noise in the first minute of elution

                df_i = df_i[(df_i.index >= min(x_range)) & (df_i.index <= max(x_range))]
                if correction_plotting_intensity and axbaseline_intensity is not None:
                    axbaseline_intensity.plot(df_i.index, df_i[col], linestyle='-')
                max_noise = abs(df_noise[col].max())# * 1.5
                threshold = max_noise 

                df_i.loc[df_i[col] < threshold, col] = replace_by
                df_i = self.keep_largest_non_nan_block(df_i, col)

            if correction_plotting_MMD and baseline is not None and threshold is not None and axbaseline_MMD is not None:
                # axbaseline_MMD.plot(df_i.index, df_i['MMD'], label = sample_name)
                baseline_filtered = baseline[(baseline.index >= min(x_range)) & (baseline.index <= max(x_range))]
                axbaseline_MMD.plot(baseline_filtered.index, baseline_filtered['y'], linestyle='--', alpha=0.5)
                axbaseline_MMD.set_xlabel(r'$\bf{Elution\ time}\ \it{(min)}$')
                axbaseline_MMD.set_ylabel(r'$\bf{MMD}\ \it{(a.u.)}$')
                axbaseline_MMD.set_title(f'MMD Raw data corrected with threshold at {threshold} of max')

            if correction_plotting_intensity and baseline is not None and threshold is not None and axbaseline_intensity is not None:
                baseline_filtered = baseline[(baseline.index >= min(x_range)) & (baseline.index <= max(x_range))]
                axbaseline_intensity.plot(df_i.index, df_i['Concentration mg/mL'], linestyle='--')
                axbaseline_intensity.plot(baseline_filtered.index, baseline_filtered['y'], linestyle='--', alpha=0.5)
                axbaseline_intensity.set_xlabel(r'$\bf{Elution\ time}\ \it{(min)}$')
                axbaseline_intensity.set_ylabel(r'$\bf{Intensity}\ \it{(mg/mL)}$')
                axbaseline_intensity.set_title(f'Raw data corrected with threshold at {threshold} of max')
                    
                axbaseline_intensity.axhline(y = threshold, color='red', linestyle='--', label = 'threshold')
            data_corrected[sample_name] = df_i
        return data_corrected

    def calculate_MMD_from_raw_data(self, plotting = False):
        """
        Calculate wlogM from raw data. It is calculated as w(logM) = intensity / |d(logM)/d(elution_volume)| and then normalized by the total sum. 
        
        Parameters
        ----------
        data_raw : dict
            Dictionary containing raw data (DataFrame) for each sample.
        
        Returns
        -------
        dict
            Dictionary containing wlogM data (DataFrame) for each sample with logM as index.
        """
        data_MMD = {}
        axintensity = None
        if plotting:
            figintensity, axintensity = plt.subplots(figsize=(6, 5))
        data_corrected = self.raw_data_correction(self.data_converted, columns_to_correct=['Concentration mg/mL'], replace_by=np.nan)
        for sample_name, df in data_corrected.items():
            # df_i = df.set_index('LogM').copy()
            # MMD_i = df_i[['MMD']].copy()
            # axintensity.plot(MMD_i['MMD'], marker='o', label = sample_name, markersize=1)
            # data_MMD[sample_name] = MMD_i

            if df.index.name != 'Elution Volume (mL)':
                df = df.set_index('Elution Volume (mL)')
            #From intensity now
            for_MMD = df[['LogM', 'Concentration mg/mL']].copy()
            #Will take the index as x axis for the gradient (derivative)
            x = pd.to_numeric(for_MMD.index, errors='coerce')
            #will now assign a new column _x , drop the NaN and sort the values according to _x
            for_MMD = for_MMD.assign(_x=x).dropna(subset=['_x']).sort_values('_x')
            
            dlogMdEV = np.gradient(for_MMD['LogM'].to_numpy(),
                                for_MMD['_x'].to_numpy())
            jac = np.abs(dlogMdEV)  # Jacobian d(LogM)/d(elution_volume)
            jac[jac == 0] = np.finfo(float).eps
            calculate = for_MMD['Concentration mg/mL'].to_numpy() / jac
            total = np.nansum(calculate)
            if total > 0:
                MMD = calculate / total
            else:
                # If total is zero or NaN, return NaNs to keep shape consistent
                MMD = np.full_like(calculate, np.nan, dtype=float)
            logM_vals = for_MMD['LogM'].to_numpy()
            df_MMD = pd.DataFrame({'MMD': MMD}, index=logM_vals)
            df_MMD.index.name = 'LogM'

            data_MMD[sample_name] = df_MMD

            if plotting and axintensity is not None:
                axintensity.plot(df_MMD, marker='o', label = sample_name, markersize=1)

        if plotting and axintensity is not None:
            axintensity.set_xlabel(r'$\bf{LogM}\ \it{(g/mol)}$')
            axintensity.set_ylabel(r'$\bf{w(logM)}\ \it{(a.u.)}$')
            axintensity.set_title('from raw data')
        return data_MMD
    
    def save_results_to_csv(self, filepath_saving):
        """
        Save results to separate CSV files:
        - One CSV per sample for MMD data
        - One CSV for all Mw/Mn results
        - One CSV per sample for raw corrected data
        
        Parameters
        ----------
        filepath_saving : str
            Directory path where to save the CSV files
        """
        # Créer le dossier s'il n'existe pas
        if not os.path.exists(filepath_saving):
            os.makedirs(filepath_saving)
        
        # Sauvegarder les MMD (un fichier par échantillon)
        mmd_folder = os.path.join(filepath_saving, 'MMD')
        if not os.path.exists(mmd_folder):
            os.makedirs(mmd_folder)
        
        for sample_name, df in self.data_MMD_all.items():
            safe_name = "".join(x for x in sample_name if x.isalnum() or x in "._- ")
            file_path = os.path.join(mmd_folder, f"MMD_{safe_name}.csv")
            df.to_csv(file_path)
        
        # Sauvegarder les Mw, Mn, PDI (un seul fichier)
        if hasattr(self, 'data_Mn_Mw'):
            mw_mn_path = os.path.join(filepath_saving, 'Mw_Mn_Results.csv')
            
            # Si le fichier existe déjà, charger les données existantes
            if os.path.exists(mw_mn_path):
                existing_data = pd.read_csv(mw_mn_path, index_col=0)
                # Combiner avec les nouvelles données (les nouvelles écrasent les anciennes pour les mêmes index)
                combined_data = pd.concat([existing_data, self.data_Mn_Mw])
                # Supprimer les duplicats en gardant la dernière occurrence (les nouvelles données)
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data.to_csv(mw_mn_path)
            else:
                # Si le fichier n'existe pas, créer un nouveau fichier
                self.data_Mn_Mw.to_csv(mw_mn_path)
        
        # Sauvegarder les données brutes corrigées (un fichier par échantillon)
        if hasattr(self, 'data_raw_corrected'):
            raw_folder = os.path.join(filepath_saving, 'Raw_Corrected')
            if not os.path.exists(raw_folder):
                os.makedirs(raw_folder)
            
            for sample_name, df in self.data_raw_corrected.items():
                safe_name = "".join(x for x in sample_name if x.isalnum() or x in "._- ")
                file_path = os.path.join(raw_folder, f"Raw_{safe_name}.csv")
                df.to_csv(file_path)

        print(f"Results saved in {filepath_saving}")

    def plotting_MMD_Mw(self, data_MMD, scale, label1 = 'Experiment', label2 = None, label3 = None,):

        """
        Plotting Molar Mass Distribution (MMD) against Molar Mass (Mw).

        Parameters
        ----------
        data_MMD : dict
            Dictionary containing MMD data (DataFrame) for each sample. This is obtained using the function extract_MMD_from_excel.
        scale : str
            Scale for the x-axis, either 'linear' or 'log'.
        write_title : bool, optional
            If True, adds a title to the figure, must add a title string as fig_title = --- . Default is False.
        fig_title : str, optional
            Title of the figure. Default is None.
        Mn_plotting : bool, optional
            If True, plots vertical lines for Mn values from self.data_Mn_Mw. Default is False.
        zoom_windows_x and zoom_windows_y : list or tuple, optional
            Two-element list or tuple containing the x-axis and y-axis limits for zooming. Default is None.
        filepath_figure_saving : str, optional
            Path where the figure will be saved. Default is None.
        fig_name_saving : str, optional
            Name of the figure file (without extension). Default is None.
        """

        fig,ax = plt.subplots(figsize=(6, 5))#, dpi=300)
        #data_MMD_all = {}
        seen_labels = set()

        for sample_name, df in data_MMD.items():
            exp_name = self.sample_information[sample_name]['Experiment']
            df_not_log = 10 ** df.index
            label_name = str(self.sample_information[sample_name][label1])
            if label2 is not None:
                label_name += f' — {str(self.sample_information[sample_name][label2])}'
            if label3 is not None:
                label_name += f' — {str(self.sample_information[sample_name][label3])}'
            if label_name not in seen_labels:
                seen_labels.add(label_name)
                if type(self.palette) == dict: 
                    ax.plot(df_not_log, df['MMD'], label = label_name, color = self.palette[self.sample_information[sample_name]['Experiment']])#, marker = 'o', markersize = 1)
                elif type(self.palette) == list:
                    ax.plot(df_not_log, df['MMD'], label = label_name, color = self.palette[len(seen_labels)-1])#, marker = 'o', markersize = 1)
            else:
                # Cas répétition du label: si palette est une liste on reprend l'indice déjà utilisé
                if type(self.palette) == dict:
                    ax.plot(df_not_log, df['MMD'], color = self.palette[self.sample_information[sample_name]['Experiment']])
                elif type(self.palette) == list:
                    # len(seen_labels)-1 correspond à l'indice attribué lors de la première apparition
                    ax.plot(df_not_log, df['MMD'], color = self.palette[len(seen_labels)-1])
        ax.set_xscale(scale)

        if scale == 'linear':
            ax.set_xlabel(r'$\bf{M}\ \it{(g/mol)}$')
        else:
            ax.set_xlabel(r'$\bf{logM}\ \it{(g/mol)}$')
        i=0
        ax.set_ylabel(r'$\bf{w(logM)}\ \it{(a.u.)}$')
        
        plt.tight_layout()
        plt.legend()#bbox_to_anchor=(1.04,0.5), loc='center left')
        plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
        
        return fig, ax

    def plotting_Mw_Mn_scatter(self, data_Mw_Mn_all, xlabel = 'Experiment', label=None, rotation=75):
        """
        Plotting scatter plots for Mw and Mn.
        """
        data_Mw_Mn_all = self.calculate_Mn_Mw_raw_data(xlabel=xlabel)

        mw_data = data_Mw_Mn_all['Mw']
        mn_data = data_Mw_Mn_all['Mn']
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8, 5))
        x_vals = []
        i=0

        for exp_name in mw_data.index:
            # get x-value from sample_information (e.g. Milling Time (s) or Experiment)
            x_val = self.sample_information[exp_name][xlabel]
            x_vals.append(x_val)
            
            # Déterminer la couleur depuis la palette
            if type(self.palette) == dict:
                color = self.palette[self.sample_information[exp_name]['Experiment']]
            elif type(self.palette) == list:
                color = self.palette[i % len(self.palette)]
            else:
                color = None  # Couleur par défaut matplotlib

            # build legend string: "Experiment — <label>"
            if label != None:
                exp_str = str(self.sample_information[exp_name].get('Experiment', exp_name))
                label_val = str(self.sample_information[exp_name].get(label, ''))
                legend_str = f"{exp_str} — {label_val}"

            # plot points; give each point its legend entry
                ax1.scatter(x_vals[i], mw_data[exp_name], label=legend_str, color=color, s=50, alpha=0.8)
                ax2.scatter(x_vals[i], mn_data[exp_name], label=legend_str, color=color, s=50, alpha=0.8)
            else:
                ax1.scatter(x_vals[i], mw_data[exp_name], color=color, s=50, alpha=0.8)
                ax2.scatter(x_vals[i], mn_data[exp_name], color=color, s=50, alpha=0.8)
            i += 1

        ax1.set_ylabel(r'$\bf{Mw}\ \it{(g/mol)}$')
        ax2.set_ylabel(r'$\bf{Mn}\ \it{(g/mol)}$')
        ax1.set_xlabel(f'{xlabel}')
        ax2.set_xlabel(f'{xlabel}')

        # set xticks using unique ordered values to avoid duplicate-tick issues
        unique_x = []
        for xv in x_vals:
            if xv not in unique_x:
                unique_x.append(xv)
        ax1.set_xticks(x_vals)
        ax1.set_xticklabels(x_vals)
        ax2.set_xticks(x_vals)
        ax2.set_xticklabels(x_vals)

        # remap plotted x positions (points were plotted using actual values),
        # ensure points align with tick indices when ticks are integer positions
        # if x_vals are already the desired numeric positions, you can skip this remapping.
        # Here we assume categorical x -> map to integer positions for display consistency
        # (only affects tick labels; points already at x_vals numeric positions).
        plt.tight_layout()
        ax1.tick_params(axis='x', labelrotation=rotation)
        ax2.tick_params(axis='x', labelrotation=rotation)
        # show legend only if label parameter is provided
        if label is not None:
            ax1.legend()
            ax2.legend()
        return fig, ax1, ax2
    
    def plotting_Mw_Mn_boxplot(self, data_Mw_Mn_all, unit='g/mol', xlabel='Experiment', label=None, rotation=75):
        """
        Plotting boxplots for Mw and Mn when multiple measurements exist per condition.
        
        This function groups data by the specified xlabel field and creates boxplots
        to show the distribution of molecular weights for each group.
        
        Parameters
        ----------
        data_Mw_Mn_all : DataFrame
            DataFrame containing Mw and Mn data for all samples.
            Should be obtained from calculate_Mn_Mw_raw_data().
            
        xlabel : str, optional
            Field to group by and use as x-axis labels.
            Can be 'Milling Time (s)', 'Experiment', 'Mass of PP (g)', etc.
            Default is 'Experiment'.
            
        label : str, optional
            Field to use for color-coding different groups (e.g., 'Beads Type').
            If None, all boxes use default colors.
            Default is None.
            
        rotation : int, optional
            Rotation angle for x-axis labels in degrees. Default is 75.
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax1 : matplotlib.axes.Axes
            Axes for Mw boxplot
        ax2 : matplotlib.axes.Axes
            Axes for Mn boxplot
            
        Examples
        --------
        >>> # Group by milling time, color by bead type
        >>> fig, ax1, ax2 = gpc.plotting_Mw_Mn_boxplot(
        ...     gpc.data_Mn_Mw,
        ...     xlabel='Milling Time (s)',
        ...     label='Beads Type'
        ... )
        
        >>> # Simple boxplot without color coding
        >>> fig, ax1, ax2 = gpc.plotting_Mw_Mn_boxplot(
        ...     gpc.data_Mn_Mw,
        ...     xlabel='Experiment',
        ...     label=None
        ... )
        """
        # Calculate data with proper grouping
        data = self.calculate_Mn_Mw_raw_data(xlabel=xlabel)
        
        # Group data by xlabel field
        mw_grouped = data.groupby(data.index.map(lambda x: self.sample_information[x][xlabel]))
        mn_grouped = data.groupby(data.index.map(lambda x: self.sample_information[x][xlabel]))
        
        mw_data = mw_grouped['Mw'].apply(list)
        mn_data = mn_grouped['Mn'].apply(list)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Utiliser la palette pour attribuer une couleur à chaque boxplot
        # On récupère les samples correspondant à chaque groupe pour trouver la couleur
        sample_groups = {}  # Pour mapper x_val -> premier sample_name rencontré
        for sample_name in data.index:
            x_val = self.sample_information[sample_name][xlabel]
            if x_val not in sample_groups:
                sample_groups[x_val] = sample_name
        
        # Plot avec les couleurs de la palette
        for i, (x_val, mw_vals) in enumerate(mw_data.items()):
            sample_name = sample_groups[x_val]
            if unit == 'kg/mol':
                mw_vals = [val / 1000 for val in mw_vals]
            
            # Déterminer la couleur depuis la palette
            if type(self.palette) == dict:
                color = self.palette[self.sample_information[sample_name]['Experiment']]
            elif type(self.palette) == list:
                color = self.palette[i % len(self.palette)]
            else:
                color = "#461ae4"  # Couleur par défaut
            
            bp1 = ax1.boxplot([mw_vals], positions=[i], widths=0.4, 
                              patch_artist=True,
                              boxprops=dict(facecolor=color, color=color, alpha=0.7),
                              medianprops=dict(color="black", linewidth=2))
            
        for i, (x_val, mn_vals) in enumerate(mn_data.items()):
            sample_name = sample_groups[x_val]
            if unit == 'kg/mol':
                mn_vals = [val / 1000 for val in mn_vals]
            
            # Déterminer la couleur depuis la palette
            if type(self.palette) == dict:
                color = self.palette[self.sample_information[sample_name]['Experiment']]
            elif type(self.palette) == list:
                color = self.palette[i % len(self.palette)]
            else:
                color = "#e41a1c"  # Couleur par défaut

            bp2 = ax2.boxplot([mn_vals], positions=[i], widths=0.4,
                              patch_artist=True,
                              boxprops=dict(facecolor=color, color=color, alpha=0.7),
                              medianprops=dict(color="black", linewidth=2))
        
        # Créer une légende si label est spécifié
        if label is not None:
            from matplotlib.patches import Patch
            # Collecter les combinaisons uniques de couleur et label
            unique_combos = {}
            for x_val in mw_data.index:
                sample_name = sample_groups[x_val]
                label_val = str(self.sample_information[sample_name].get(label, 'Unknown'))
                
                if type(self.palette) == dict:
                    color = self.palette[self.sample_information[sample_name]['Experiment']]
                elif type(self.palette) == list:
                    idx = list(mw_data.index).index(x_val)
                    color = self.palette[idx % len(self.palette)]
                else:
                    color = "#461ae4"
                
                if label_val not in unique_combos:
                    unique_combos[label_val] = color
            
            legend_elements = [Patch(facecolor=color, label=lbl, alpha=0.7) 
                             for lbl, color in unique_combos.items()]
            ax1.legend(handles=legend_elements, title=label)
            ax2.legend(handles=legend_elements, title=label)
        
        # Set labels and ticks
        ax1.set_ylabel(rf'$\bf{{Mw}}\ \it{{({unit})}}$')
        ax1.set_xlabel(xlabel)
        ax1.set_xticks(np.arange(len(mw_data)))
        ax1.set_xticklabels(mw_data.index, rotation=rotation)

        ax2.set_ylabel(rf'$\bf{{Mn}}\ \it{{({unit})}}$')
        ax2.set_xlabel(xlabel)
        ax2.set_xticks(np.arange(len(mn_data)))
        ax2.set_xticklabels(mn_data.index, rotation=rotation)
        
        plt.tight_layout()
        return fig, ax1, ax2
    """
    If the report used is the 'excel report', hence use the processed and calculated MMD vs LogM and Mw, Mn from the software directly 
    (will be more dependant on the baseline choosen by the user on the software)
    """
    def extract_Mw_Mn_from_excel(self):
        data_Mw_Mn_all = {}

        for file, path in self.filepath_dict.items():
            data_Mw_Mn_i = pd.read_excel(path, sheet_name='Results')

            Mw_i = float(data_Mw_Mn_i.loc[0,"Unnamed: 6"])  # type: ignore
            Mn_i = float(data_Mw_Mn_i.loc[1,"Unnamed: 6"])  # type: ignore
            PDI_i = Mw_i/Mn_i
            data_Mw_Mn_all[file] = [Mw_i, Mn_i, PDI_i]

        if not data_Mw_Mn_all:
            print(f"No files found in filepath_dict")
            print(f"Available files: {self.filepath_dict.keys()}")
            return pd.DataFrame()  # Return an empty DataFrame
        data_Mw_Mn_all = pd.DataFrame(data_Mw_Mn_all).T
        data_Mw_Mn_all.columns = ['Mw', 'Mn', 'PDI']
        return data_Mw_Mn_all

    def extract_MMD_from_excel(self):
        data_MMD_all = {}
        for file, path in self.filepath_dict.items():
            data_MMD_i = pd.read_excel(path, sheet_name='Data MMD', header=0, index_col=0)
            data_MMD_i = pd.DataFrame(data_MMD_i['MMD'], columns=['MMD'])
            # Post-processing MMD data: clamp negatives to 0
            data_MMD_i['MMD'] = data_MMD_i['MMD'].clip(lower=0)
            data_MMD_all[file] = data_MMD_i
        return data_MMD_all
    
    def calculate_Mn_Mw_from_MMD(self, data_MMD_all):
        """Calculate Mw, Mn, PDI, and M_max from MMD data as intensity = MMD.
        Parameters
        ----------
        data_MMD_all : dict
            Dictionary containing MMD data (DataFrame) for each sample."""
        Mn_Mw_from_MMD = {}        
        for sample_name, df in data_MMD_all.items():
            intensity = df['MMD']
            
            Mi = 10 ** df.index  # Convert logM to M
            # Mw = sum(Mi**2 *intensity) / sum(Mi*intensity)
            if sum(intensity) == 0 or sum(intensity/Mi) == 0:
                print(f"Warning: Sum of intensity or sum of intensity/Mi is zero for sample {sample_name}. Setting Mw, Mn, PDI, M_max to NaN.")
                Mw = np.nan
                Mn = np.nan
                PDI = np.nan
                M_max = np.nan
            else:
                Mw = sum(Mi*intensity) / sum(intensity)  # Weight-average molar mass
                Mn = sum(intensity) / sum(intensity/Mi)  # Number-average molar mass
                M_max = 10 ** intensity.idxmax() 
                PDI = Mw/Mn
            Mn_Mw_from_MMD[sample_name] = [Mw, Mn, PDI, M_max]
        Mn_Mw_from_MMD = pd.DataFrame(Mn_Mw_from_MMD).T
        Mn_Mw_from_MMD.columns = ['Mw', 'Mn', 'PDI', 'M_max']

        return Mn_Mw_from_MMD