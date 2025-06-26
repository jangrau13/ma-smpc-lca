import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import ast
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PATHS AS NEEDED
# =============================================================================

# Manually specify CSV file paths
CSV_FILE_PATHS = [
    "multi-server/fixed-point/results/fixed_point_results.csv",
    "multi-server/floating-point/results/floating_point_results.csv",
    "single-server/cpu-results.csv",
    "single-server/gpu-results.csv",
    # Add more CSV file paths as needed
]

# Output folder for generated plots
OUTPUT_FOLDER = 'analysis_results'  # Change this to your preferred folder name

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================

# Create output directory for figures
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def detect_metric_type(df):
    """
    Detect whether a DataFrame contains fixed-point, floating-point, or mixed metrics.
    
    Returns: 'fixed', 'float', 'mixed', or 'unknown'
    """
    fixed_cols = [col for col in df.columns if col.startswith('fixed_')]
    float_cols = [col for col in df.columns if col.startswith('float_')]
    
    has_fixed = len(fixed_cols) > 0
    has_float = len(float_cols) > 0
    
    if has_fixed and has_float:
        return 'mixed'
    elif has_fixed:
        return 'fixed'
    elif has_float:
        return 'float'
    else:
        return 'unknown'

def standardize_column_names(df, metric_type):
    """
    Standardize column names based on the detected metric type.
    For files with only one type, we'll rename columns to include the prefix.
    """
    if metric_type == 'mixed':
        return df  # Already has proper prefixes
    
    # Column mappings for common metric names
    metric_mappings = {
        'max_abs_error': 'max_abs_error',
        'max_rel_error': 'max_rel_error', 
        'mean_abs_error': 'mean_abs_error',
        'mean_rel_error': 'mean_rel_error',
        'snr_db': 'snr_db',
        'norm_error': 'norm_error'
    }
    
    df_copy = df.copy()
    
    if metric_type in ['fixed', 'float']:
        prefix = f"{metric_type}_"
        
        # Rename columns that match our known metrics
        for original_suffix, standard_suffix in metric_mappings.items():
            # Try different possible column names
            possible_names = [original_suffix, standard_suffix]
            
            for possible_name in possible_names:
                if possible_name in df_copy.columns:
                    new_name = f"{prefix}{standard_suffix}"
                    if possible_name != new_name:
                        df_copy = df_copy.rename(columns={possible_name: new_name})
                    break
    
    return df_copy

def load_and_combine_csvs(file_paths):
    """
    Load multiple CSV files and combine them into a single DataFrame.
    Handles separate fixed-point and floating-point files intelligently.
    """
    dataframes = []
    file_info = []
    
    # First pass: Load all files and detect their types
    for i, file_path in enumerate(file_paths):
        try:
            print(f"Loading CSV file {i+1}/{len(file_paths)}: {file_path}")
            df_temp = pd.read_csv(file_path)
            
            # Detect what type of metrics this file contains
            metric_type = detect_metric_type(df_temp)
            
            # Standardize column names
            df_temp = standardize_column_names(df_temp, metric_type)
            
            # Add metadata
            df_temp['source_file'] = os.path.basename(file_path)
            df_temp['source_index'] = i
            df_temp['source_path'] = file_path
            df_temp['metric_type'] = metric_type
            
            dataframes.append(df_temp)
            file_info.append({
                'path': file_path,
                'type': metric_type,
                'shape': df_temp.shape,
                'columns': list(df_temp.columns)
            })
            
            print(f"  Successfully loaded {len(df_temp)} rows from {file_path}")
            print(f"  Detected metric type: {metric_type}")
            
        except FileNotFoundError:
            print(f"  WARNING: File not found: {file_path} - Continuing with other files...")
            continue
        except pd.errors.EmptyDataError:
            print(f"  WARNING: Empty file: {file_path} - Continuing with other files...")
            continue
        except Exception as e:
            print(f"  WARNING: Error loading {file_path}: {str(e)} - Continuing with other files...")
            continue
    
    if not dataframes:
        raise ValueError("No CSV files were successfully loaded. Please check your file paths.")
    
    print(f"\n=== FILE ANALYSIS SUMMARY ===")
    for info in file_info:
        print(f"File: {info['path']}")
        print(f"  Type: {info['type']}, Shape: {info['shape']}")
        
    # Strategy for combining: 
    # 1. If we have separate fixed/float files, try to merge them on common keys
    # 2. Otherwise, just concatenate everything
    
    fixed_dfs = [df for df in dataframes if df['metric_type'].iloc[0] == 'fixed']
    float_dfs = [df for df in dataframes if df['metric_type'].iloc[0] == 'float'] 
    mixed_dfs = [df for df in dataframes if df['metric_type'].iloc[0] == 'mixed']
    
    combined_dfs = []
    
    # Add mixed dataframes as-is
    combined_dfs.extend(mixed_dfs)
    
    # Try to merge fixed and float dataframes
    if fixed_dfs and float_dfs:
        print(f"\n=== MERGING STRATEGY ===")
        print(f"Found {len(fixed_dfs)} fixed-point file(s) and {len(float_dfs)} floating-point file(s)")
        
        # Combine all fixed dataframes
        if len(fixed_dfs) > 1:
            fixed_combined = pd.concat(fixed_dfs, ignore_index=True)
        else:
            fixed_combined = fixed_dfs[0]
            
        # Combine all float dataframes  
        if len(float_dfs) > 1:
            float_combined = pd.concat(float_dfs, ignore_index=True)
        else:
            float_combined = float_dfs[0]
        
        # Find common columns for merging (excluding metric columns and metadata)
        exclude_cols = ['source_file', 'source_index', 'source_path', 'metric_type']
        fixed_non_metric_cols = [col for col in fixed_combined.columns 
                               if not col.startswith(('fixed_', 'float_')) and col not in exclude_cols]
        float_non_metric_cols = [col for col in float_combined.columns 
                               if not col.startswith(('fixed_', 'float_')) and col not in exclude_cols]
        
        common_cols = list(set(fixed_non_metric_cols) & set(float_non_metric_cols))
        
        if common_cols:
            print(f"Merging on common columns: {common_cols}")
            
            # Merge on common experimental parameters
            merged_df = pd.merge(fixed_combined, float_combined, 
                               on=common_cols, how='outer', 
                               suffixes=('_fixed_src', '_float_src'))
            
            # Clean up metadata columns after merge
            if 'source_file_fixed_src' in merged_df.columns:
                merged_df['source_file'] = merged_df['source_file_fixed_src'].fillna(merged_df['source_file_float_src'])
                merged_df.drop(['source_file_fixed_src', 'source_file_float_src'], axis=1, inplace=True)
            
            combined_dfs.append(merged_df)
            print(f"Successfully merged fixed and float data. Result shape: {merged_df.shape}")
        else:
            print("No common columns found for merging. Concatenating separately.")
            combined_dfs.extend(fixed_dfs)
            combined_dfs.extend(float_dfs)
    else:
        # Add remaining dataframes
        combined_dfs.extend(fixed_dfs)
        combined_dfs.extend(float_dfs)
    
    # Final combination
    if len(combined_dfs) > 1:
        final_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
    else:
        final_df = combined_dfs[0]
    
    print(f"\nFinal combined dataset shape: {final_df.shape}")
    print(f"Available columns: {list(final_df.columns)}")
    
    return final_df

# --- 1. Data Loading and Preprocessing ---
print("=" * 50)
print("MULTI-CSV DATA ANALYSIS SCRIPT")
print("=" * 50)
print("\n--- 1. Data Loading and Preprocessing ---")

# Load and combine CSV files
try:
    df = load_and_combine_csvs(CSV_FILE_PATHS)
except Exception as e:
    print(f"ERROR combining CSV files: {str(e)}")
    exit(1)

# Display basic information about the combined dataset
print(f"\nDataset Info:")
print(f"  Shape: {df.shape}")
print(f"  Data types summary:")
print(df.dtypes.value_counts().to_string())

# Only dropping seed and matrix_cond_T as requested initially
columns_to_ignore = ['seed', 'matrix_cond_T']

# For consistency in analysis, we can use a specific precision or all.
# The original script filtered to precision 30. We will retain this if the column exists.
if 'PRECISION' in df.columns:
    print(f"\nAvailable PRECISION values: {sorted(df['PRECISION'].unique())}")
    df = df[df['PRECISION'] == 30]
    print(f"Filtered to PRECISION == 30. Remaining rows: {len(df)}")
else:
    print("\nNo PRECISION column found. Using all data.")

df.drop(columns=columns_to_ignore, inplace=True, errors='ignore')
print(f"Dropped ignored columns: {columns_to_ignore}")

# Define a function to clean stringified lists and take their mean
def clean_and_mean(value):
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        try:
            list_val = ast.literal_eval(value)
            if isinstance(list_val, list) and all(isinstance(i, (int, float)) for i in list_val):
                return np.mean(list_val)
        except (ValueError, SyntaxError):
            return value
    return value

# Apply the cleaning function to object columns that are not simple booleans
print("\nCleaning stringified list columns...")
for col in df.columns:
    if df[col].dtype == 'object':
        if not df[col].astype(str).str.lower().isin(['true', 'false']).all():
            df[col] = df[col].apply(clean_and_mean)

# Handle boolean-like columns, like 'INTRODUCE_OUTLIERS'
if 'INTRODUCE_OUTLIERS' in df.columns:
    if df['INTRODUCE_OUTLIERS'].dtype == 'object':
        df['INTRODUCE_OUTLIERS'] = df['INTRODUCE_OUTLIERS'].str.lower().map({'true': 1, 'false': 0})
    elif pd.api.types.is_bool_dtype(df['INTRODUCE_OUTLIERS']):
        df['INTRODUCE_OUTLIERS'] = df['INTRODUCE_OUTLIERS'].astype(int)

# Convert all columns to numeric, coercing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(axis=1, how='all', inplace=True)

print(f"Data cleaning complete. Final shape: {df.shape}")

# Check what metric columns we actually have
fixed_cols = [col for col in df.columns if col.startswith('fixed_')]
float_cols = [col for col in df.columns if col.startswith('float_')]
print(f"\nAvailable fixed-point metrics: {fixed_cols}")
print(f"Available floating-point metrics: {float_cols}")

# --- 1.5: Create Bucketing Columns ---
print("\n--- 1.5. Creating Bucketing Columns for Grouped Analysis ---")
if 'num_outliers_injected' in df.columns:
    if (df['num_outliers_injected'] == 0).any():
        df.loc[df['num_outliers_injected'] == 0, 'outlier_bucket'] = 'Zero'
        non_zero_mask = df['num_outliers_injected'] > 0
        if non_zero_mask.sum() > 3:
            try:
                df.loc[non_zero_mask, 'outlier_bucket'] = pd.qcut(df.loc[non_zero_mask, 'num_outliers_injected'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            except ValueError:
                df.loc[non_zero_mask, 'outlier_bucket'] = pd.cut(df.loc[non_zero_mask, 'num_outliers_injected'], bins=3, labels=['Low', 'Medium', 'High'], include_lowest=True)
        elif non_zero_mask.sum() > 0:
            df.loc[non_zero_mask, 'outlier_bucket'] = 'High'
    else:
        if df['num_outliers_injected'].nunique() > 3:
            df['outlier_bucket'] = pd.qcut(df['num_outliers_injected'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
        else:
            df['outlier_bucket'] = df['num_outliers_injected'].astype(str)
    print("Created 'outlier_bucket' column.")

if 'matrix_cond_A' in df.columns:
    df['log_cond_A'] = np.log10(df['matrix_cond_A'].astype(float) + 1e-9)
    if df['log_cond_A'].nunique() > 3:
        df['cond_bucket'] = pd.qcut(df['log_cond_A'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    else:
        df['cond_bucket'] = 'Default'
    print("Created 'cond_bucket' column from 'matrix_cond_A'.")

# Create relative outlier metric for feature importance analysis
if 'num_outliers_injected' in df.columns and 'N' in df.columns:
    df['num_outliers_injected_rel'] = df['num_outliers_injected'] / (df['N'] * df['N'])
else:
    if 'num_outliers_injected_rel' not in df.columns: 
        df['num_outliers_injected_rel'] = 0

print("-" * 50)

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS (SECTION 2)
# =============================================================================

# --- 2. Comparative Feature Importance Analysis (Float vs Fixed Point) ---
print("\n--- 2. FEATURE IMPORTANCE ANALYSIS ---")
print("--- 2.1 Comparative Feature Importance Analysis (Float vs Fixed Point) ---")

base_factors = ['N', 'R', 'INTRODUCE_OUTLIERS', 'OUTLIER_PROBABILITY', 'B_INT_RANGE_SCALE', 'A_DENSITY', 'matrix_cond_A', 'num_outliers_injected_rel', 'MINIMUM_NOISE_RANGE_VAL']
float_factors = [f for f in base_factors if f in df.columns]
fixed_factors = [f for f in (base_factors + ['PRECISION']) if f in df.columns]

print(f"Floating Point Factors: {float_factors}")
print(f"Fixed Point Factors: {fixed_factors}")

# Build comparative metrics based on what's available
comparative_metrics = {}
potential_metrics = [
    ('Max Absolute Error', 'max_abs_error'),
    ('Max Relative Error', 'max_rel_error'),
    ('Mean Absolute Error', 'mean_abs_error'),
    ('Mean Relative Error', 'mean_rel_error'),
    ('SNR (dB)', 'snr_db')
]

for display_name, base_name in potential_metrics:
    fixed_col = f'fixed_{base_name}'
    float_col = f'float_{base_name}'
    
    if fixed_col in df.columns and float_col in df.columns:
        comparative_metrics[display_name] = (float_col, fixed_col)
    elif fixed_col in df.columns:
        print(f"  Note: Only fixed-point data available for {display_name}")
    elif float_col in df.columns:
        print(f"  Note: Only floating-point data available for {display_name}")

print(f"Comparative Metric Pairs: {list(comparative_metrics.keys())}")
comparative_results = {}

plot_counter = 1

for metric_name, (float_col, fixed_col) in comparative_metrics.items():
    if float_col not in df.columns or fixed_col not in df.columns:
        continue
    metric_results = {}
    for impl_name, (col_name, factors) in {'Float': (float_col, float_factors), 'Fixed': (fixed_col, fixed_factors)}.items():
        temp_df = df[[col_name] + factors].dropna()
        if temp_df.empty or temp_df[col_name].nunique() < 2: continue
        X = temp_df[factors].loc[:, temp_df[factors].nunique() > 1]
        if X.empty: continue
        y = temp_df[col_name]
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        correlations = X.corrwith(y).fillna(0)
        metric_results[impl_name] = pd.DataFrame({'Importance': importances, 'Correlation': correlations})
    if metric_results: comparative_results[metric_name] = metric_results

print("\n--- Creating Comparative Feature Importance Plots (Sorted by Fixed-Point)---")
if comparative_results:
    for metric_name, implementations in comparative_results.items():
        if len(implementations) == 2:
            float_data, fixed_data = implementations.get('Float'), implementations.get('Fixed')
            if float_data is not None and fixed_data is not None:
                all_factors = float_data.index.union(fixed_data.index)
                comp_df = pd.DataFrame(index=all_factors)
                comp_df['Float_Imp'] = float_data['Importance']
                comp_df['Fixed_Imp'] = fixed_data['Importance']
                comp_df['Float_Corr'] = float_data['Correlation']
                comp_df['Fixed_Corr'] = fixed_data['Correlation']
                comp_df.fillna(0, inplace=True)
                comp_df = comp_df.sort_values(by='Fixed_Imp', ascending=True)

                plt.figure(figsize=(15, 8))
                x_pos = np.arange(len(comp_df))
                width = 0.35
                float_colors = ['#2ca02c' if c > 0 else '#d62728' for c in comp_df['Float_Corr']]
                fixed_colors = ['#2ca02c' if c > 0 else '#d62728' for c in comp_df['Fixed_Corr']]

                plt.barh(x_pos - width/2, comp_df['Float_Imp'], width, color=float_colors, alpha=0.8, label='Floating Point', edgecolor='black', linewidth=1, hatch='///')
                plt.barh(x_pos + width/2, comp_df['Fixed_Imp'], width, color=fixed_colors, alpha=0.8, label='Fixed Point', edgecolor='black', linewidth=1)

                plt.yticks(x_pos, [f.upper() for f in comp_df.index], fontsize=11)
                plt.xlabel('Feature Importance (Random Forest)', fontsize=14)
                plt.ylabel('Factors', fontsize=14)
                plt.title(f'Comparative Feature Importance: {metric_name}', fontsize=16, weight='bold', pad=20)

                legend_elements = [
                    mpatches.Patch(facecolor='#2ca02c', edgecolor='black', hatch='///', label='FLOATING (POSITIVE CORR.)'),
                    mpatches.Patch(facecolor='#d62728', edgecolor='black', hatch='///', label='FLOATING (NEGATIVE CORR.)'),
                    mpatches.Patch(facecolor='#2ca02c', edgecolor='black', hatch=None, label='FIXED (POSITIVE CORR.)'),
                    mpatches.Patch(facecolor='#d62728', edgecolor='black', hatch=None, label='FIXED (NEGATIVE CORR.)')
                ]
                plt.legend(handles=legend_elements, loc='lower right', fontsize=10, ncol=2)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()

                metric_clean = metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
                filename = f'{OUTPUT_FOLDER}/{plot_counter:02d}_{metric_clean}_float_vs_fixed_comparison.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"Saved: {filename}")
                plot_counter += 1
else:
    print("Could not perform comparative feature importance analysis.")

# --- 2.2. Side-by-Side Feature Importance: MRE vs SNR ---
print("\n--- 2.2. Side-by-Side Feature Importance: MRE vs SNR ---")
if 'comparative_results' in locals() and \
   'Mean Relative Error' in comparative_results and \
   'SNR (dB)' in comparative_results:

    mre_data = comparative_results['Mean Relative Error']
    snr_data = comparative_results['SNR (dB)']

    all_factors_unioned = set()
    if 'Float' in mre_data: all_factors_unioned.update(mre_data['Float'].index)
    if 'Fixed' in mre_data: all_factors_unioned.update(mre_data['Fixed'].index)
    if 'Float' in snr_data: all_factors_unioned.update(snr_data['Float'].index)
    if 'Fixed' in snr_data: all_factors_unioned.update(snr_data['Fixed'].index)

    sorting_df = pd.DataFrame(index=list(all_factors_unioned))
    if 'Fixed' in mre_data:
        sorting_df['mre_fixed_imp'] = mre_data['Fixed']['Importance']
    if 'Fixed' in snr_data:
        sorting_df['snr_fixed_imp'] = snr_data['Fixed']['Importance']
    sorting_df.fillna(0, inplace=True)
    sorting_df['total_fixed_imp'] = sorting_df.get('mre_fixed_imp', 0) + sorting_df.get('snr_fixed_imp', 0)
    sorted_factors = sorting_df.sort_values('total_fixed_imp', ascending=True).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(22, 12), sharey=True)
    fig.suptitle('Comparative Feature Importance: Mean Relative Error vs. SNR (dB)\n(Sorted by Combined Fixed-Point Importance)',
                 fontsize=22, weight='bold', y=0.98)

    plot_configs = {
        'Mean Relative Error': {'data': mre_data, 'ax': axes[0]},
        'SNR (dB)': {'data': snr_data, 'ax': axes[1]}
    }

    for metric_name, config in plot_configs.items():
        ax = config['ax']
        implementations = config['data']
        if len(implementations) >= 2:
            float_data, fixed_data = implementations.get('Float'), implementations.get('Fixed')
            if float_data is not None and fixed_data is not None:
                comp_df = pd.DataFrame(index=sorted_factors)
                comp_df['Float_Imp'] = float_data['Importance']
                comp_df['Fixed_Imp'] = fixed_data['Importance']
                comp_df['Float_Corr'] = float_data['Correlation']
                comp_df['Fixed_Corr'] = fixed_data['Correlation']
                comp_df.fillna(0, inplace=True)

                x_pos = np.arange(len(comp_df))
                width = 0.4
                float_colors = ['#2ca02c' if c > 0 else '#d62728' for c in comp_df['Float_Corr']]
                fixed_colors = ['#2ca02c' if c > 0 else '#d62728' for c in comp_df['Fixed_Corr']]

                ax.barh(x_pos - width/2, comp_df['Float_Imp'], width, color=float_colors, alpha=0.8, label='Floating Point', edgecolor='black', linewidth=1, hatch='///')
                ax.barh(x_pos + width/2, comp_df['Fixed_Imp'], width, color=fixed_colors, alpha=0.8, label='Fixed Point', edgecolor='black', linewidth=1)

                ax.set_xlabel('Feature Importance (Random Forest)', fontsize=16)
                if metric_name == 'Mean Relative Error':
                    ax.set_ylabel('Factors', fontsize=16)
                ax.set_title(f'Importance for {metric_name}', fontsize=18, weight='bold', pad=20)
                ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"Data not available for\n{metric_name}", ha='center', va='center', fontsize=14, style='italic')

    plt.yticks(np.arange(len(sorted_factors)), [f.upper() for f in sorted_factors], fontsize=12)

    legend_elements = [
        mpatches.Patch(facecolor='grey', edgecolor='black', hatch='///', label='FLOATING POINT'),
        mpatches.Patch(facecolor='grey', edgecolor='black', hatch=None,  label='FIXED POINT'),
        mpatches.Patch(color='#2ca02c', label='Positive Correlation with Metric'),
        mpatches.Patch(color='#d62728', label='Negative Correlation with Metric')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=4, fontsize=12, fancybox=True, shadow=True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    filename = f'{OUTPUT_FOLDER}/{plot_counter:02d}_mre_vs_snr_feature_importance_sorted_by_fixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")
    plot_counter += 1
else:
    print("Could not generate side-by-side MRE vs SNR feature importance plot.")
    print("Required data from 'comparative_results' not found.")

print("-" * 50)

# =============================================================================
# PRECISION ANALYSIS (SECTION 3)
# =============================================================================

# --- 3. Fixed-Point Metric Comparison vs. N by PRECISION ---
print("\n--- 3. PRECISION ANALYSIS ---")
print("--- 3.1 Fixed-Point Error Plots by PRECISION ---")
if 'PRECISION' in df.columns:
    # Build available fixed-point metrics
    fixed_metrics_to_plot = {}
    for display_name, base_name in potential_metrics:
        fixed_col = f'fixed_{base_name}'
        if fixed_col in df.columns:
            fixed_metrics_to_plot[display_name] = fixed_col
    
    if fixed_metrics_to_plot:
        precision_values = sorted(df['PRECISION'].unique())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fixed-Point Error Metrics vs. Matrix Size (N) by Precision',
                     fontsize=20, weight='bold', y=0.97)
        axes_flat = axes.flatten()
        colors = plt.cm.viridis(np.linspace(0, 1, len(precision_values)))

        for idx, (title, fix_col) in enumerate(fixed_metrics_to_plot.items()):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]

            for i, prec in enumerate(precision_values):
                prec_df = df[df['PRECISION'] == prec]
                if not prec_df.empty:
                    grouped_data = prec_df.groupby('N')[fix_col].mean().reset_index()
                    if not grouped_data.empty and not grouped_data[fix_col].isnull().all():
                        ax.plot(grouped_data['N'], grouped_data[fix_col], marker='o',
                                label=f'P={prec}', color=colors[i], linewidth=2.5, markersize=7)

            ax.set_title(title, fontsize=14, weight='bold', pad=15)
            ax.set_xlabel('Matrix Size (N)', fontsize=12)
            if 'Error' in title:
                ax.set_yscale('log')
                ax.set_ylabel('Value (Log Scale)', fontsize=12)
            else:
                ax.set_ylabel('Value (dB)', fontsize=12)
            ax.legend(title='Precision', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, which="both", ls="--", alpha=0.6)

        # Remove unused subplots
        for idx in range(len(fixed_metrics_to_plot), len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        filename = f'{OUTPUT_FOLDER}/{plot_counter:02d}_fixed_error_metrics_by_precision.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")
        plot_counter += 1
    else:
        print("No fixed-point metrics found for precision analysis.")
else:
    print("Column 'PRECISION' not found.")

print("-" * 50)

# =============================================================================
# ERROR COMPARISON ANALYSIS (SECTION 4)
# =============================================================================

# --- 4. Error Comparison Analysis ---
print("\n--- 4. ERROR COMPARISON ANALYSIS ---")

# Define metrics for comparison based on what's available
metrics_to_plot = comparative_metrics.copy()

# --- 4.1. Individual Metric Comparison Plots vs. N ---
print("\n--- 4.1 Individual Metric Comparison Plots vs. N ---")

if metrics_to_plot:
    columns_for_grouping = list(sum(metrics_to_plot.values(), ()))
    available_columns = [col for col in columns_for_grouping if col in df.columns]
    
    if available_columns and 'N' in df.columns:
        grouped_df = df.groupby('N')[available_columns].mean().reset_index()

        for title, (float_col, fix_col) in metrics_to_plot.items():
            plt.figure(figsize=(10, 6))
            plt.style.use('seaborn-v0_8-whitegrid')

            if fix_col in grouped_df.columns:
                plt.plot(grouped_df['N'], grouped_df[fix_col], marker='o', linestyle='-',
                        label=f'Fixed Point {title}', linewidth=2.5, markersize=8, color='#1f77b4')
            if float_col in grouped_df.columns:
                plt.plot(grouped_df['N'], grouped_df[float_col], marker='s', linestyle='--',
                        label=f'Floating Point {title}', linewidth=2.5, markersize=7, color='#ff7f0e')

            plt.title(f'{title} Comparison vs. Matrix Size (N)', fontsize=16, weight='bold', pad=20)
            plt.xlabel('Matrix Size (N)', fontsize=14)

            if 'Error' in title:
                plt.yscale('log')
                plt.ylabel('Average Value (Log Scale)', fontsize=14)
            else:
                plt.ylabel('Value (dB)', fontsize=14)

            plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
            plt.grid(True, which="both", ls="--", alpha=0.6)
            plt.tight_layout()

            metric_name = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            plt.savefig(f'{OUTPUT_FOLDER}/{plot_counter:02d}_{metric_name}_vs_N.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"Saved: {plot_counter:02d}_{metric_name}_vs_N.png")
            plot_counter += 1

        print("Generated and saved individual plots for metric comparison relative to N.")
    else:
        print("Insufficient data for individual metric plots.")
else:
    print("No comparative metrics available for plotting.")

# --- 4.2: Group-wise Comparison Grids ---
print("\n--- 4.2: Group-wise Comparison Grids ---")

if metrics_to_plot:
    grouping_setups = {
        'A_DENSITY': {'col': 'A_DENSITY', 'name': 'Matrix Density'},
        'Outliers': {'col': 'outlier_bucket', 'name': 'Number of Outliers'},
        'Condition': {'col': 'cond_bucket', 'name': 'Matrix Condition (κ(A))'}
    }

    for key, setup in grouping_setups.items():
        group_col = setup['col']
        group_name = setup['name']
        print(f"\n--- Generating Comparison Grid for: {group_name} ---")

        if group_col not in df.columns:
            print(f"Column '{group_col}' not found. Skipping grid.")
            continue

        if group_col == 'outlier_bucket':
            order = ['Zero', 'Low', 'Medium', 'High']
            unique_groups = sorted(df[group_col].dropna().unique(), key=lambda x: order.index(x) if x in order else len(order))
        else:
            unique_groups = sorted(df[group_col].dropna().unique())

        if not unique_groups:
            print(f"No unique groups found for {group_name}. Skipping grid.")
            continue

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.suptitle(f'Error Metrics Comparison vs. N, Grouped by {group_name}',
                     fontsize=22, weight='bold', y=1.0)
        axes_flat = axes.flatten()

        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_groups)))

        for idx, (title, (float_col, fix_col)) in enumerate(metrics_to_plot.items()):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]

            for i, group_val in enumerate(unique_groups):
                subset_df = df[df[group_col] == group_val]
                if subset_df.empty:
                    continue

                grouped_data = subset_df.groupby('N')[[fix_col, float_col]].mean().reset_index()

                if fix_col in grouped_data.columns and not grouped_data[fix_col].isnull().all():
                    ax.plot(grouped_data['N'], grouped_data[fix_col], marker='o', linestyle='-',
                            color=colors[i], linewidth=2.5, markersize=6)

                if float_col in grouped_data.columns and not grouped_data[float_col].isnull().all():
                    ax.plot(grouped_data['N'], grouped_data[float_col], marker='x', linestyle='--',
                            color=colors[i], linewidth=2.5, markersize=6)

            ax.set_title(title, fontsize=16, weight='bold', pad=15)
            ax.set_xlabel('Matrix Size (N)', fontsize=14)
            if 'Error' in title:
                ax.set_yscale('log')
                ax.set_ylabel('Value (Log Scale)', fontsize=14)
            else:
                ax.set_ylabel('Value (dB)', fontsize=14)
            ax.grid(True, which="both", ls="--", alpha=0.6)

        color_patches = [mpatches.Patch(color=colors[i], label=f'{unique_groups[i]}') for i in range(len(unique_groups))]
        style_lines = [Line2D([0], [0], color='grey', linestyle='-', marker='o', label='Fixed-Point'),
                       Line2D([0], [0], color='grey', linestyle='--', marker='x', label='Floating-Point')]

        # Remove unused subplots and add legend
        used_plots = min(len(metrics_to_plot), len(axes_flat))
        if used_plots < len(axes_flat):
            legend_ax = axes_flat[-1]
            legend_ax.axis('off')
            leg1 = legend_ax.legend(handles=color_patches, loc='upper left', title=f'{group_name} Buckets', fontsize=12, title_fontsize=14)
            legend_ax.add_artist(leg1)
            legend_ax.legend(handles=style_lines, loc='lower left', title='Data Type', fontsize=12, title_fontsize=14)
            
            # Remove other unused subplots
            for idx in range(used_plots, len(axes_flat) - 1):
                fig.delaxes(axes_flat[idx])
        else:
            fig.legend(handles=color_patches + style_lines, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        filename = f'{OUTPUT_FOLDER}/{plot_counter:02d}_error_metrics_by_{key.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")
        plot_counter += 1


print("=" * 50)
print("ANALYSIS COMPLETE!")
print(f"Generated {plot_counter - 1} visualization files in '{OUTPUT_FOLDER}' directory.")
loaded_files = [path for path in CSV_FILE_PATHS if os.path.exists(path)]
missing_files = [path for path in CSV_FILE_PATHS if not os.path.exists(path)]
print(f"Successfully processed {len(loaded_files)} CSV files:")
for path in loaded_files:
    print(f"  ✓ {path}")
if missing_files:
    print(f"Missing files ({len(missing_files)}):")
    for path in missing_files:
        print(f"  ✗ {path}")
print("=" * 50)