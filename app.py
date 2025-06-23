import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime, timedelta
import random

# Set page configuration
st.set_page_config(
    page_title="CohortManager",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)



def create_availability_heatmap(df, field_colors, site_info, actual_data_df=None):
    """Create an interactive heatmap showing data availability with characteristic colors"""
    
    # Determine if we're using real cohort data or dummy data
    if len(site_info) == len(df.columns):
        # Real data: extract cohorts from unique case IDs in DataFrame columns
        # DataFrame columns are like "cohort1_case123", so extract the cohort part
        column_cohorts = [col.split('_')[0] for col in df.columns]
        unique_sites = sorted(list(set(column_cohorts)))  # Sort for consistent ordering
        sites = unique_sites
        
        # Reorder DataFrame columns by cohort for proper visual grouping
        ordered_columns = []
        for site in sites:
            site_patients = [col for col in df.columns if col.startswith(f"{site}_")]
            site_patients.sort()  # Sort within each cohort for consistency
            ordered_columns.extend(site_patients)
        
        # Reorder the DataFrame
        df = df[ordered_columns]
        if actual_data_df is not None:
            actual_data_df = actual_data_df[ordered_columns]
        
        # Now calculate site positions based on the reordered DataFrame
        site_positions = {}
        current_pos = 0
        
        for site in sites:
            # Find all columns that belong to this cohort in the reordered DataFrame
            site_patients = [col for col in df.columns if col.startswith(f"{site}_")]
            if site_patients:
                site_positions[site] = (current_pos, current_pos + len(site_patients) - 1)
                current_pos += len(site_patients)
    else:
        # Dummy data: use original logic
        sites = ['Site_A', 'Site_B', 'Site_C']
        site_positions = {}
        current_pos = 0
        
        for site in sites:
            site_patients = [col for col in df.columns if col.startswith(site)]
            if site_patients:
                site_positions[site] = (current_pos, current_pos + len(site_patients) - 1)
                current_pos += len(site_patients)
    
    # Create custom hover text for all cells
    hover_text = []
    for i, field in enumerate(df.index):
        row_hover = []
        for j, val in enumerate(df.iloc[i]):
            patient = df.columns[j]
            # Extract cohort and case_id from unique case ID
            if '_' in patient:
                cohort = patient.split('_')[0]
                case_id = '_'.join(patient.split('_')[1:])  # Join in case case_id has underscores
                patient_display = f"{case_id} (Cohort: {cohort})"
            else:
                patient_display = patient
            status = "Available" if val == 1 else "Missing"
            row_hover.append(f'<b>Field:</b> {field}<br><b>Patient:</b> {patient_display}<br><b>Status:</b> {status}')
        hover_text.append(row_hover)
    
    # Create a single heatmap with custom colors using RGB mapping
    # We'll map field types to different RGB values and create a custom colorscale
    
    # Create a color-coded matrix where each field gets a unique color range
    field_list = list(df.index)
    color_matrix = np.zeros_like(df.values, dtype=float)
    
    # Assign color codes based on field types
    field_type_codes = {}
    unique_field_types = []
    
    for field in field_list:
        if 'WSI' in field or 'Slide' in field:
            field_type = 'sample'
        else:
            field_type = 'clinical'
        
        if field_type not in unique_field_types:
            unique_field_types.append(field_type)
        field_type_codes[field] = unique_field_types.index(field_type)
    
    # Create the color matrix
    for i, field in enumerate(field_list):
        base_code = field_type_codes[field] * 10  # Separate field types by 10
        color_matrix[i] = np.where(df.iloc[i] == 1, base_code + 1, base_code)  # +1 for available, +0 for missing
    
    # Create colorscale for different field types
    if 'sample' in unique_field_types and 'clinical' in unique_field_types:
        # Both clinical and sample data
        colorscale = [
            [0.0, '#000000'],    # Clinical missing (black)
            [0.125, '#1f77b4'],  # Clinical available (blue)
            [0.25, '#000000'],   # Gap
            [0.875, '#000000'],  # Sample missing (black)  
            [1.0, '#9467bd']     # Sample available (purple)
        ]
    else:
        # Only clinical data
        colorscale = [
            [0.0, '#000000'],    # Missing (black)
            [1.0, '#1f77b4']     # Available (blue)
        ]
    
    # Create the figure
    fig = go.Figure(data=go.Heatmap(
        z=color_matrix,
        x=df.columns,
        y=df.index,
        colorscale=colorscale,
        showscale=False,
        hoverongaps=False,
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text
    ))
    
    # Add vertical lines to separate sites
    shapes = []
    for i, site in enumerate(sites[:-1]):  # Don't add line after last site
        if site in site_positions:
            x_pos = site_positions[site][1] + 0.5
            shapes.append(dict(
                type="line",
                x0=x_pos, x1=x_pos,
                y0=-0.5, y1=len(df.index) - 0.5,
                line=dict(color="white", width=4)
            ))
    
    fig.update_layout(
        title={
            'text': 'Clinical Data Availability Matrix by Site',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(
            title="Patients (grouped by site)",
            showticklabels=False,  # Hide case ID labels to reduce clutter
            side='bottom'
        ),
        yaxis=dict(
            title="Clinical Fields",
            autorange='reversed',  # Keep fields in original order
            dtick=1,  # Force one tick per field
            tickmode='linear'  # Ensure linear spacing
        ),
        width=1400,
        height=max(600, 30 * len(df.index)),  # Dynamic height based on number of fields
        margin=dict(l=200, r=100, t=100, b=50),
        shapes=shapes
    )
    
    # Add site annotations
    for site, (start, end) in site_positions.items():
        fig.add_annotation(
            x=(start + end) / 2,
            y=len(df.index),
            text=f"<b>{site}</b>",
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
    
    return fig

def calculate_statistics(df):
    """Calculate statistics about data availability"""
    
    # Overall statistics
    total_cells = df.size
    available_cells = df.sum().sum()
    missing_cells = total_cells - available_cells
    
    # Per patient statistics
    patient_completeness = df.sum(axis=0) / len(df) * 100
    
    # Per field statistics
    field_completeness = df.sum(axis=1) / len(df.columns) * 100
    
    return {
        'total_cells': total_cells,
        'available_cells': available_cells,
        'missing_cells': missing_cells,
        'overall_completeness': (available_cells / total_cells) * 100,
        'patient_completeness': patient_completeness,
        'field_completeness': field_completeness
    }

def load_sample_mapping_data(mapping_path, sample_type="Sample", clinical_df=None):
    """Load sample mapping data (e.g., WSI slides) and return case_id mapping with cohort awareness"""
    
    try:
        # Load the mapping CSV file
        mapping_df = pd.read_csv(mapping_path)
        
        # Check if we have case_id column
        if 'case_id' not in mapping_df.columns:
            st.error(f"Sample mapping file must contain 'case_id' column")
            return None, None, None
        
        # Convert case_id to string for consistency
        mapping_df['case_id'] = mapping_df['case_id'].astype(str)
        
        # Check if the mapping file already has cohort information
        if 'cohort' in mapping_df.columns:
            # Use the cohort information directly from the mapping file (more reliable)
            # Create unique identifier combining cohort and case_id
            mapping_df['unique_case_id'] = mapping_df['cohort'].astype(str) + '_' + mapping_df['case_id'].astype(str)
            
            # Group by unique case ID
            groupby_column = 'unique_case_id'
            
            # Also create mapping from unique_case_id back to original case_id for display
            unique_to_original = dict(zip(mapping_df['unique_case_id'], mapping_df['case_id']))
            
            st.sidebar.info(f"ℹ️ {sample_type}: Using cohort info from mapping file")
            
        elif clinical_df is not None and 'cohort' in clinical_df.columns:
            # Fallback: Create a lookup dictionary for case_id -> cohort from clinical data
            # Note: This can be problematic if case_ids are not unique across cohorts
            clinical_df['case_id'] = clinical_df['case_id'].astype(str)
            case_to_cohort = dict(zip(clinical_df['case_id'], clinical_df['cohort']))
            
            # Add cohort information to mapping data
            mapping_df['cohort'] = mapping_df['case_id'].map(case_to_cohort)
            
            # Create unique identifier combining cohort and case_id
            mapping_df['unique_case_id'] = mapping_df['cohort'].astype(str) + '_' + mapping_df['case_id'].astype(str)
            
            # Group by unique case ID
            groupby_column = 'unique_case_id'
            
            # Also create mapping from unique_case_id back to original case_id for display
            unique_to_original = dict(zip(mapping_df['unique_case_id'], mapping_df['case_id']))
            
            st.sidebar.warning(f"⚠️ {sample_type}: Using cohort mapping from clinical data (may have conflicts)")
            
        else:
            # Fallback to original case_id if no cohort info available
            mapping_df['unique_case_id'] = mapping_df['case_id']
            groupby_column = 'case_id'
            unique_to_original = dict(zip(mapping_df['case_id'], mapping_df['case_id']))
            
            st.sidebar.warning(f"⚠️ {sample_type}: No cohort info found, using case_id only")
        
        # Group by the appropriate column to count samples per case
        sample_counts = mapping_df.groupby(groupby_column).size().reset_index(name='sample_count')
        sample_counts.rename(columns={groupby_column: 'unique_case_id'}, inplace=True)
        
        # Also store the actual sample IDs for each case
        sample_details = mapping_df.groupby(groupby_column).apply(
            lambda x: x.iloc[:, 1].tolist() if len(x.columns) > 1 else []
        ).to_dict()
        
        return sample_counts, sample_details, mapping_df, unique_to_original
        
    except Exception as e:
        st.error(f"Error loading sample mapping file: {str(e)}")
        return None, None, None, None

def integrate_sample_data(availability_df, actual_df, field_colors, sample_mappings):
    """Integrate sample data into the clinical availability matrix"""
    
    if not sample_mappings:
        return availability_df, actual_df, field_colors
    
    # Create a copy to avoid modifying original data
    new_availability_df = availability_df.copy()
    new_actual_df = actual_df.copy()
    new_field_colors = field_colors.copy()
    
    # Define colors for different sample types
    sample_colors = {
        'WSI_Slides': '#9B59B6',          # Purple - WSI
        'Proteomics_Samples': '#E67E22',   # Orange - Proteomics
        'Genomics_Samples': '#27AE60',     # Green - Genomics
        'Pathology_Samples': '#F39C12',    # Yellow - Pathology
        'Sample_Count': '#8E44AD'          # Dark Purple - Count
    }
    
    for sample_type, (sample_counts, sample_details, raw_mapping, unique_to_original) in sample_mappings.items():
        # Create availability row for this sample type
        sample_availability = []
        sample_values = []
        
        for case_id in availability_df.columns:
            # Try to find samples using the unique case ID (which might be cohort_case_id format)
            case_samples = sample_counts[sample_counts['unique_case_id'] == case_id]
            if not case_samples.empty and case_samples['sample_count'].iloc[0] > 0:
                sample_availability.append(1)
                sample_values.append(int(case_samples['sample_count'].iloc[0]))
            else:
                sample_availability.append(0)
                sample_values.append(0)
        
        # Add to availability matrix
        sample_field_name = f"{sample_type}_Available"
        new_availability_df.loc[sample_field_name] = sample_availability
        new_actual_df.loc[sample_field_name] = sample_values
        
        # Add color for this sample type
        if sample_type in sample_colors:
            new_field_colors[sample_field_name] = sample_colors[sample_type]
        else:
            # Generate a color if not predefined
            new_field_colors[sample_field_name] = f"#{random.randint(0,255):02x}{random.randint(0,255):02x}{random.randint(0,255):02x}"
    
    return new_availability_df, new_actual_df, new_field_colors

def load_real_csv_data(csv_path, subsample_size=None, sample_mappings_config=None):
    """Load real clinical data from CSV file and convert to availability matrix format"""
    
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Subsample if requested
        if subsample_size and subsample_size < len(df):
            df = df.sample(n=subsample_size, random_state=42).reset_index(drop=True)
        
        # Extract case IDs and cohort information
        case_ids = df['case_id'].astype(str)
        cohorts = df['cohort']
        
        # Create unique case identifiers combining cohort and case_id to handle duplicate case_ids across cohorts
        unique_case_ids = cohorts.astype(str) + '_' + case_ids.astype(str)
        
        # Get clinical fields (exclude case_id and cohort columns)
        clinical_fields = [col for col in df.columns if col not in ['case_id', 'cohort']]
        
        # Create availability matrix (fields as rows, patients as columns)
        availability_matrix = []
        actual_data = {}
        
        # Define colors for different clinical field categories
        field_colors = {
            'FIGO_stage': '#FF6B6B',           # Red - Staging
            'Grade': '#FF8E8E',                # Light Red - Grading
            'HRD_score': '#4ECDC4',           # Teal - Molecular
            'BRCA_status': '#6DD4CC',         # Light Teal - Genetics
            'Proteomics_PTMs': '#45B7D1',     # Blue - Proteomics
            'Transcriptomics_PTMs': '#6BC4DA', # Light Blue - Transcriptomics
            'Plat_response': '#96CEB4',       # Green - Treatment Response
            'Neoadjuvant_treatment': '#A8D5BC' # Light Green - Treatment
        }
        
        for field in clinical_fields:
            # Check availability for each patient (1 if not null/empty, 0 if null/empty)
            field_data = df[field]
            availability = []
            values = []
            
            for i, val in enumerate(field_data):
                # Consider data available if it's not NaN, not empty string, and not just whitespace
                if pd.isna(val) or str(val).strip() == '' or str(val).strip().lower() == 'nan':
                    availability.append(0)
                    values.append(np.nan)
                else:
                    availability.append(1)
                    values.append(val)
            
            availability_matrix.append(availability)
            actual_data[field] = values
        
        # Convert to DataFrames using unique case IDs as columns
        availability_df = pd.DataFrame(availability_matrix, index=clinical_fields, columns=unique_case_ids)
        actual_df = pd.DataFrame(actual_data, index=unique_case_ids).T
        
        # Load and integrate sample mapping data
        sample_mappings = {}
        if sample_mappings_config:
            for sample_type, mapping_path in sample_mappings_config.items():
                if mapping_path and os.path.exists(mapping_path):
                    result = load_sample_mapping_data(mapping_path, sample_type, df)  # Pass clinical df for cohort-aware merging
                    if result[0] is not None:
                        sample_counts, sample_details, raw_mapping, unique_to_original = result
                        sample_mappings[sample_type] = (sample_counts, sample_details, raw_mapping, unique_to_original)
                        
                        # Show detailed merge statistics
                        total_samples = len(raw_mapping)
                        unique_cases = len(sample_counts)
                        
                        # Use cohort information from the processed mapping data
                        if 'cohort' in raw_mapping.columns:
                            cohort_breakdown = raw_mapping['cohort'].value_counts().to_dict()
                        else:
                            cohort_breakdown = {"Unknown": total_samples}
                        
                        success_msg = f"✅ Loaded {sample_type}: {total_samples} samples for {unique_cases} unique patients"
                        cohort_details = ", ".join([f"{k}:{v}" for k, v in cohort_breakdown.items()])
                        st.sidebar.success(f"{success_msg}")
                        st.sidebar.info(f"📊 Sample distribution: {cohort_details}")
                        
                        # Debug: Show specific cohort merge success by counting unique case IDs
                        for cohort, sample_count in cohort_breakdown.items():
                            cohort_cases = sample_counts[sample_counts['unique_case_id'].str.startswith(f"{cohort}_")]
                            st.sidebar.info(f"🔗 {cohort}: {len(cohort_cases)} patients with {sample_type}")
        
        # Integrate sample data into availability matrix
        if sample_mappings:
            availability_df, actual_df, field_colors = integrate_sample_data(
                availability_df, actual_df, field_colors, sample_mappings
            )
        
        # Create site information from cohorts
        site_info = cohorts.tolist()
        
        return availability_df, actual_df, field_colors, site_info, cohorts.unique(), sample_mappings
        
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None, None, None, None, None

def main():
    # Header
    st.markdown('<div class="main-header">🏥 CohortManager</div>', unsafe_allow_html=True)
    st.markdown("### Clinical Data Availability Dashboard")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Data loading - Real CSV Data
    st.sidebar.subheader("📁 Clinical Data")
    
    # Check if user wants directory loading instead
    use_directory = st.sidebar.checkbox(
        "Load from Directory (experimental)", 
        value=False, 
        help="Load data from directory structure instead of CSV files"
    )
    
    if not use_directory:
        csv_path = st.sidebar.text_input(
            "CSV File Path",
            value="/tank/WSI_data/Ovarian_WSIs/OV_master/master_cohort.csv",
            help="Path to the CSV file containing clinical data"
        )
        
        # Sample mapping configuration
        st.sidebar.subheader("📊 Sample Data Integration")
        
        # Enable sample mapping
        enable_sample_mapping = st.sidebar.checkbox(
            "Include Sample Data",
            value=True,
            help="Load and integrate sample mapping data (e.g., WSI slides, proteomics samples)"
        )
        
        sample_mappings_config = {}
        if enable_sample_mapping:
            # WSI Slides mapping
            wsi_mapping_path = st.sidebar.text_input(
                "WSI Slides Mapping CSV",
                value="/tank/WSI_data/Ovarian_WSIs/OV_master/master_WSI_mappings.csv",
                help="Path to CSV mapping case_id to WSI slides"
            )
            if wsi_mapping_path.strip():
                sample_mappings_config["WSI_Slides"] = wsi_mapping_path
            
            # Add more sample types
            with st.sidebar.expander("➕ Additional Sample Types", expanded=False):
                # Proteomics samples
                proteomics_path = st.sidebar.text_input(
                    "Proteomics Samples CSV",
                    value="",
                    placeholder="/path/to/proteomics_mapping.csv",
                    help="Path to CSV mapping case_id to proteomics samples"
                )
                if proteomics_path.strip():
                    sample_mappings_config["Proteomics_Samples"] = proteomics_path
                
                # Genomics samples
                genomics_path = st.sidebar.text_input(
                    "Genomics Samples CSV",
                    value="",
                    placeholder="/path/to/genomics_mapping.csv",
                    help="Path to CSV mapping case_id to genomics samples"
                )
                if genomics_path.strip():
                    sample_mappings_config["Genomics_Samples"] = genomics_path
                
                # Custom sample type
                custom_name = st.sidebar.text_input(
                    "Custom Sample Type Name",
                    value="",
                    placeholder="e.g., Pathology_Samples",
                    help="Name for custom sample type"
                )
                custom_path = st.sidebar.text_input(
                    "Custom Sample Type CSV",
                    value="",
                    placeholder="/path/to/custom_mapping.csv",
                    help="Path to CSV for custom sample type"
                )
                if custom_name.strip() and custom_path.strip():
                    sample_mappings_config[custom_name] = custom_path
        
        # Add subsample option for testing
        subsample_data = st.sidebar.checkbox(
            "Subsample for testing",
            value=False,
            help="Load only a subset of patients to test visualization performance"
        )
        
        if subsample_data:
            subsample_size = st.sidebar.slider(
                "Number of patients to load",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Reduce the number of patients for testing"
            )
        else:
            subsample_size = None
        
        # Debug option to clear cache
        if st.sidebar.button("🗑️ Clear Cache & Reload", help="Force reload all data and clear cache"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.sidebar.success("Cache cleared! Data will reload.")
            st.rerun()
        
        if csv_path and os.path.exists(csv_path):
            st.sidebar.success("CSV file found!")
            
            # Check if we need to load/reload the data
            config_changed = (
                'cohort_data' not in st.session_state or 
                'csv_path' not in st.session_state or 
                st.session_state.get('csv_path') != csv_path or 
                st.session_state.get('subsample_size') != subsample_size or
                st.session_state.get('sample_mappings_config') != sample_mappings_config
            )
            
            if config_changed or st.sidebar.button("Reload CSV Data"):
                
                with st.spinner("Loading real clinical data and sample mappings..."):
                    # Load real CSV data with sample mappings
                    result = load_real_csv_data(csv_path, subsample_size, sample_mappings_config)
                    
                    if result[0] is not None:  # Check if loading was successful
                        availability_df, actual_df, field_colors, site_info, cohorts, sample_mappings = result
                        st.session_state.cohort_data = availability_df
                        st.session_state.actual_data = actual_df
                        st.session_state.field_colors = field_colors
                        st.session_state.site_info = site_info
                        st.session_state.cohorts = cohorts
                        st.session_state.sample_mappings = sample_mappings
                        st.session_state.csv_path = csv_path
                        st.session_state.subsample_size = subsample_size
                        st.session_state.sample_mappings_config = sample_mappings_config
                        
                        # Count clinical vs sample fields
                        clinical_fields = [f for f in availability_df.index if not f.endswith('_Available')]
                        sample_fields = [f for f in availability_df.index if f.endswith('_Available')]
                        
                        sample_text = f" (subsampled)" if subsample_size else ""
                        success_msg = f"✅ Loaded {len(availability_df.columns)} patients{sample_text}"
                        success_msg += f" with {len(clinical_fields)} clinical fields"
                        if sample_fields:
                            success_msg += f" + {len(sample_fields)} sample types"
                        st.sidebar.success(success_msg)
                    else:
                        st.sidebar.error("Failed to load CSV data")
                        # Fallback to dummy data
                        availability_df, actual_df, field_colors, site_info = generate_dummy_data(20)
                        st.session_state.cohort_data = availability_df
                        st.session_state.actual_data = actual_df
                        st.session_state.field_colors = field_colors
                        st.session_state.site_info = site_info
                        st.session_state.sample_mappings = {}
            
            # Use cached data
            df = st.session_state.cohort_data
            actual_data_df = st.session_state.actual_data
            field_colors = st.session_state.field_colors
            site_info = st.session_state.site_info
            
        else:
            st.sidebar.error("❌ Please provide a valid CSV file path to continue")
            st.stop()  # Stop execution until valid path is provided
            
    else:
        # Directory loading (experimental)
        data_directory = st.sidebar.text_input(
            "Data Directory Path",
            placeholder="/path/to/clinical/data",
            help="Path to directory containing clinical data files"
        )
        
        if data_directory and os.path.exists(data_directory):
            st.sidebar.success("Directory found!")
            st.sidebar.info("Directory loading not yet implemented. Please use CSV file loading.")
            st.stop()
        else:
            st.sidebar.error("❌ Please provide a valid directory path")
            st.stop()
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=len(df.columns)
        )
    
    with col2:
        st.metric(
            label="Clinical Fields",
            value=len(df.index)
        )
    
    with col3:
        # Count patients per site/cohort
        if 'site_info' in st.session_state and len(st.session_state.site_info) == len(df.columns):
            # Real data: extract cohorts from DataFrame column names (unique case IDs)
            site_counts = {}
            for col in df.columns:
                cohort = col.split('_')[0]  # Extract cohort from unique case ID
                site_counts[cohort] = site_counts.get(cohort, 0) + 1
            
            # Create delta string for real cohorts
            delta_parts = [f"{k}:{v}" for k, v in sorted(site_counts.items())]
            delta_str = ", ".join(delta_parts)
        else:
            # Dummy data: use original logic
            site_counts = {}
            for col in df.columns:
                site = col.split('_')[0] + '_' + col.split('_')[1]
                site_counts[site] = site_counts.get(site, 0) + 1
            delta_str = f"A:{site_counts.get('Site_A', 0)}, B:{site_counts.get('Site_B', 0)}, C:{site_counts.get('Site_C', 0)}"
        
        st.metric(
            label="Sites/Cohorts",
            value=len(site_counts),
            delta=delta_str
        )
    
    # Main visualization
    st.subheader("Data Availability Matrix")
    
    # Display options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Field Color Legend")
        
        # Show clinical and sample fields separately
        if 'field_colors' in st.session_state:
            clinical_fields = [f for f in df.index if not f.endswith('_Available')]
            sample_fields = [f for f in df.index if f.endswith('_Available')]
            
            if clinical_fields:
                st.write("**Clinical Fields:**")
                for field in clinical_fields:
                    if field in field_colors:
                        st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 20px; background-color: {field_colors[field]}; margin-right: 10px; border: 1px solid #ccc;"></div>{field}</div>', unsafe_allow_html=True)
            
            if sample_fields:
                st.write("**Sample Data Types:**")
                for field in sample_fields:
                    if field in field_colors:
                        display_name = field.replace('_Available', '').replace('_', ' ')
                        st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 20px; background-color: {field_colors[field]}; margin-right: 10px; border: 1px solid #ccc;"></div>{display_name}</div>', unsafe_allow_html=True)
        else:
            st.info("Load clinical data to see field legend")
    
    with col1:
        # Create and display the heatmap
        if df is not None and field_colors is not None and site_info is not None:
            fig = create_availability_heatmap(df, field_colors, site_info, actual_data_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Figure creation returned None")
        else:
            st.error("Missing data for visualization. Please check data loading.")
    
    # Cohort Builder
    st.subheader("🔬 Cohort Builder")
    st.write("Select clinical fields and sample types to create a cohort of patients with complete data for **ALL** selected criteria:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Separate clinical fields from sample fields
        clinical_fields = [f for f in df.index if not f.endswith('_Available')]
        sample_fields = [f for f in df.index if f.endswith('_Available')]
        
        # Create tabs for different selection types
        tab1, tab2, tab3 = st.tabs(["🩺 Clinical Data", "🔬 Sample Data", "📋 Combined Selection"])
        
        with tab1:
            selected_clinical_fields = st.multiselect(
                "Select Clinical Fields:",
                options=clinical_fields,
                default=[],
                help="Choose clinical data fields for cohort filtering",
                key="clinical_fields_select"
            )
        
        with tab2:
            if sample_fields:
                selected_sample_fields = st.multiselect(
                    "Select Sample Types:",
                    options=sample_fields,
                    default=[],
                    help="Choose sample types that must be available for each patient",
                    key="sample_fields_select"
                )
                
                # Show sample details if available
                if 'sample_mappings' in st.session_state and st.session_state.sample_mappings:
                    with st.expander("📊 Sample Data Summary", expanded=False):
                        for sample_type, (sample_counts, sample_details, raw_mapping, unique_to_original) in st.session_state.sample_mappings.items():
                            total_samples = len(raw_mapping)
                            cases_with_samples = len(sample_counts)
                            avg_samples_per_case = sample_counts['sample_count'].mean()
                            
                            st.write(f"**{sample_type}:**")
                            st.write(f"• Total samples: {total_samples}")
                            st.write(f"• Cases with samples: {cases_with_samples}")
                            st.write(f"• Avg samples per case: {avg_samples_per_case:.1f}")
                            st.write("---")
            else:
                st.info("No sample data loaded. Enable 'Include Sample Data' in the sidebar to add sample types.")
                selected_sample_fields = []
        
        with tab3:
            # Combined selection
            all_available_fields = clinical_fields + sample_fields
            selected_fields = st.multiselect(
                "Select All Fields (Clinical + Sample):",
                options=all_available_fields,
                default=[],
                help="Choose from both clinical fields and sample types for comprehensive filtering",
                key="combined_fields_select"
            )
            
            # Merge selections from individual tabs
            if 'selected_clinical_fields' in locals() and 'selected_sample_fields' in locals():
                combined_from_tabs = selected_clinical_fields + selected_sample_fields
                if combined_from_tabs and not selected_fields:
                    selected_fields = combined_from_tabs
                    st.info(f"Auto-combined {len(selected_clinical_fields)} clinical + {len(selected_sample_fields)} sample fields")
        
        # Use the selected fields for cohort building
        if 'selected_fields' not in locals():
            selected_fields = []
        
        if selected_fields:
            # Filter patients who have data for ALL selected fields
            # For each selected field, get patients who have data (value = 1)
            cohort_patients = None
            
            for field in selected_fields:
                field_patients = set(df.columns[df.loc[field] == 1])
                if cohort_patients is None:
                    cohort_patients = field_patients
                else:
                    cohort_patients = cohort_patients.intersection(field_patients)
            
            cohort_patients = list(cohort_patients)
            cohort_patients.sort()
            
            # Create filtered dataset
            if cohort_patients:
                filtered_df = df[cohort_patients]
                
                st.success(f"✅ Found **{len(cohort_patients)}** patients with complete data for all selected fields!")
                
                # Show cohort composition by site/cohort
                site_breakdown = {}
                if 'site_info' in st.session_state and len(st.session_state.site_info) == len(df.columns):
                    # Real data: extract cohorts from unique case IDs
                    for patient in cohort_patients:
                        cohort = patient.split('_')[0]  # Extract cohort from unique case ID
                        site_breakdown[cohort] = site_breakdown.get(cohort, 0) + 1
                else:
                    # Dummy data: use original logic
                    for patient in cohort_patients:
                        site = patient.split('_')[0] + '_' + patient.split('_')[1]
                        site_breakdown[site] = site_breakdown.get(site, 0) + 1
                
                st.write("**Cohort by Site/Study:**")
                for site, count in sorted(site_breakdown.items()):
                    st.write(f"• {site}: {count} patients")
                
                # Show sample data summary for the cohort
                sample_selected = [f for f in selected_fields if f.endswith('_Available')]
                if sample_selected and 'sample_mappings' in st.session_state:
                    st.write("**Sample Data in Cohort:**")
                    for sample_field in sample_selected:
                        sample_type = sample_field.replace('_Available', '')
                        if sample_type in st.session_state.sample_mappings:
                            sample_counts, sample_details, raw_mapping, unique_to_original = st.session_state.sample_mappings[sample_type]
                            
                            # Count samples for cohort patients
                            cohort_sample_count = 0
                            for patient in cohort_patients:
                                patient_samples = sample_counts[sample_counts['unique_case_id'] == patient]
                                if not patient_samples.empty:
                                    cohort_sample_count += patient_samples['sample_count'].iloc[0]
                            
                            st.write(f"• {sample_type}: {cohort_sample_count} total samples across {len(cohort_patients)} patients")
                
                # Generate structured filename
                clinical_part = "_".join([f for f in selected_fields if not f.endswith('_Available')]).replace(" ", "")
                sample_part = "_".join([f.replace('_Available', '') for f in selected_fields if f.endswith('_Available')]).replace(" ", "")
                
                filename_parts = []
                if clinical_part:
                    filename_parts.append(f"Clinical_{clinical_part}")
                if sample_part:
                    filename_parts.append(f"Samples_{sample_part}")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                suggested_filename = f"Cohort_{'_'.join(filename_parts)}_{timestamp}" if filename_parts else f"Cohort_{timestamp}"
                
                st.write("**Selected Patients:**")
                patients_display = ", ".join(cohort_patients[:10])
                if len(cohort_patients) > 10:
                    patients_display += f"... (+{len(cohort_patients) - 10} more)"
                st.write(patients_display)
                
                # Optional Manual Field Labeling Section
                with st.expander("🏷️ Manual Field Labeling (Optional)", expanded=False):
                    st.write("Transform categorical values to numeric labels for machine learning:")
                    
                    # Initialize field mappings in session state
                    if 'field_mappings' not in st.session_state:
                        st.session_state.field_mappings = {}
                    
                    # Select field to relabel
                    relabel_field = st.selectbox(
                        "Select field to relabel:",
                        options=["Select a field..."] + selected_fields,
                        help="Choose a field to map its values to numeric labels"
                    )
                    
                    if relabel_field != "Select a field...":
                        # Get unique values for this field in the filtered cohort
                        field_data = actual_data_df.loc[relabel_field, cohort_patients].dropna()
                        unique_values = sorted(field_data.unique()) if len(field_data) > 0 else []
                        
                        if unique_values:
                            st.write(f"**Current values in '{relabel_field}':**")
                            
                            # Create mapping interface
                            field_key = f"{relabel_field}_{len(cohort_patients)}"  # Unique key
                            if field_key not in st.session_state.field_mappings:
                                st.session_state.field_mappings[field_key] = {}
                            
                            mapping_changed = False
                            col_map1, col_map2 = st.columns(2)
                            
                            with col_map1:
                                st.write("**Original Values:**")
                                for val in unique_values:
                                    st.write(f"• {val}")
                            
                            with col_map2:
                                st.write("**Map to:**")
                                for i, val in enumerate(unique_values):
                                    current_mapping = st.session_state.field_mappings[field_key].get(val, i)
                                    new_mapping = st.number_input(
                                        f"Map '{val}' to:",
                                        value=current_mapping,
                                        key=f"mapping_{field_key}_{val}",
                                        help=f"Numeric value for '{val}'"
                                    )
                                    
                                    if new_mapping != st.session_state.field_mappings[field_key].get(val):
                                        st.session_state.field_mappings[field_key][val] = new_mapping
                                        mapping_changed = True
                            
                            # Show preview of mapping
                            if st.session_state.field_mappings[field_key]:
                                st.write("**Mapping Preview:**")
                                mapping_preview = " | ".join([f"{k}→{v}" for k, v in st.session_state.field_mappings[field_key].items()])
                                st.code(mapping_preview)
                                
                                # Quick preset buttons
                                col_preset1, col_preset2, col_preset3 = st.columns(3)
                                with col_preset1:
                                    if st.button("📊 Binary (0,1)", help="Map first value to 0, second to 1"):
                                        if len(unique_values) >= 2:
                                            st.session_state.field_mappings[field_key] = {unique_values[0]: 0, unique_values[1]: 1}
                                            st.rerun()
                                
                                with col_preset2:
                                    if st.button("🔢 Sequential (0,1,2...)", help="Map values to sequential numbers"):
                                        st.session_state.field_mappings[field_key] = {val: i for i, val in enumerate(unique_values)}
                                        st.rerun()
                                
                                with col_preset3:
                                    if st.button("🧹 Clear Mapping", help="Reset all mappings"):
                                        st.session_state.field_mappings[field_key] = {}
                                        st.rerun()
                        else:
                            st.warning(f"No data available for field '{relabel_field}' in selected cohort.")
                
            else:
                st.warning("⚠️ No patients found with complete data for all selected fields.")
                filtered_df = None
                suggested_filename = None
    
    with col2:
        if selected_fields and cohort_patients:
            st.subheader("📥 Export Cohort")
            
            # Summary statistics
            st.metric("Total Patients", len(cohort_patients))
            st.metric("Data Fields", len(selected_fields))
            
            # Show active field mappings
            active_mappings = {k: v for k, v in st.session_state.get('field_mappings', {}).items() if v}
            if active_mappings:
                st.metric("Active Mappings", len(active_mappings), help="Number of fields with custom label mappings")
            
            # Custom filename input
            custom_filename = st.text_input(
                "Filename:",
                value=suggested_filename,
                help="Customize the filename for your cohort export"
            )
            
            # Export format and download in one step
            col_csv, col_json, col_excel = st.columns(3)
            
            # Create the export dataframe - EXPLODED so each sample gets its own row
            export_rows = []
            
            # Get clinical and sample field selections
            clinical_fields_selected = [f for f in selected_fields if not f.endswith('_Available')]
            sample_fields_selected = [f for f in selected_fields if f.endswith('_Available')]
            
            for case_id in cohort_patients:
                # Get clinical data for this case
                case_clinical_data = {}
                for field in clinical_fields_selected:
                    if field in actual_data_df.index:
                        case_clinical_data[field] = actual_data_df.loc[field, case_id]
                
                # If sample fields are selected, explode by samples
                if sample_fields_selected and 'sample_mappings' in st.session_state:
                    case_has_samples = False
                    
                    for sample_field in sample_fields_selected:
                        sample_type = sample_field.replace('_Available', '')
                        if sample_type in st.session_state.sample_mappings:
                            sample_counts, sample_details, raw_mapping, unique_to_original = st.session_state.sample_mappings[sample_type]
                            
                            # Get all samples for this case
                            case_samples = raw_mapping[raw_mapping['unique_case_id'] == case_id]
                            
                            if not case_samples.empty:
                                case_has_samples = True
                                # Create one row per sample
                                for _, sample_row in case_samples.iterrows():
                                    row_data = {'case_id': case_id}
                                    row_data.update(case_clinical_data)  # Add clinical data
                                    row_data[f'{sample_type}_ID'] = sample_row.iloc[0]  # First column is sample ID
                                    
                                    # Add other sample metadata if available
                                    if len(sample_row) > 2:  # More than case_id and sample_id
                                        for col_idx, col_name in enumerate(case_samples.columns):
                                            if col_idx > 1:  # Skip case_id and sample_id
                                                row_data[f'{sample_type}_{col_name}'] = sample_row.iloc[col_idx]
                                    
                                    export_rows.append(row_data)
                    
                    # If case has no samples but was selected, add one row with empty sample fields
                    if not case_has_samples:
                        row_data = {'case_id': case_id}
                        row_data.update(case_clinical_data)
                        for sample_field in sample_fields_selected:
                            sample_type = sample_field.replace('_Available', '')
                            row_data[f'{sample_type}_ID'] = ""
                        export_rows.append(row_data)
                
                else:
                    # No sample fields selected, just add clinical data
                    row_data = {'case_id': case_id}
                    row_data.update(case_clinical_data)
                    export_rows.append(row_data)
            
            # Create DataFrame from exploded rows
            export_df = pd.DataFrame(export_rows)
            
            # Set case_id as index if no samples, otherwise keep it as a column for clarity
            if not sample_fields_selected:
                export_df.set_index('case_id', inplace=True)
                export_df.index.name = 'case_id'
            
            # Apply field mappings if any exist (for clinical fields only)
            for field_key, mapping in st.session_state.get('field_mappings', {}).items():
                if mapping:  # Only if mapping is not empty
                    field_name = field_key.split('_')[0]  # Extract field name from key
                    if field_name in export_df.columns and len(mapping) > 0:
                        # Apply the mapping
                        export_df[field_name] = export_df[field_name].map(mapping).fillna(export_df[field_name])
            
            with col_csv:
                # Export exploded format - include index only if no samples (case_id is column when samples present)
                include_index = not sample_fields_selected
                csv_content = export_df.to_csv(index=include_index)
                st.download_button(
                    label="📄 CSV",
                    data=csv_content,
                    file_name=f"{custom_filename}.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col_json:
                # Use records format - reset index if needed
                if sample_fields_selected:
                    json_content = export_df.to_json(orient='records', indent=2)
                else:
                    json_content = export_df.reset_index().to_json(orient='records', indent=2)
                
                st.download_button(
                    label="📋 JSON",
                    data=json_content,
                    file_name=f"{custom_filename}.json",
                    mime="application/json"
                )
            
            with col_excel:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    # Export exploded format
                    include_index = not sample_fields_selected
                    export_df.to_excel(writer, sheet_name='Cohort_Data', index=include_index)
                excel_content = buffer.getvalue()
                
                st.download_button(
                    label="📊 Excel",
                    data=excel_content,
                    file_name=f"{custom_filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Show preview of exported data
            with st.expander("📋 Data Preview", expanded=False):
                if sample_fields_selected:
                    st.write("**Exploded Export Format** (One Row Per Sample):")
                    st.write(f"Total rows in export: **{len(export_df)}** (includes one row per sample)")
                    
                    # Count unique cases vs total rows
                    if 'case_id' in export_df.columns:
                        unique_cases = export_df['case_id'].nunique()
                        st.write(f"Unique cases: **{unique_cases}**, Total samples/rows: **{len(export_df)}**")
                    
                    # Show sample distribution
                    sample_cols = [col for col in export_df.columns if col.endswith('_ID') and not col == 'case_id']
                    if sample_cols:
                        st.write("**Sample Distribution:**")
                        for col in sample_cols:
                            non_empty = export_df[col].replace('', pd.NA).notna().sum()
                            st.write(f"• {col}: {non_empty} samples with IDs")
                else:
                    st.write("**Standard Export Format** (One Row Per Case):")
                    st.write(f"Total cases in export: **{len(export_df)}**")
                
                st.write("**Preview (first 10 rows):**")
                # Show first 10 rows and limit columns if too many
                preview_df = export_df.head(10)
                if len(export_df.columns) > 8:
                    st.write(f"Showing first 8 of {len(export_df.columns)} columns...")
                    preview_df = preview_df.iloc[:, :8]
                st.dataframe(preview_df)
    
    # Data export options
    st.subheader("Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export as CSV"):
            csv = df.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"cohort_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Statistics"):
            stats_json = json.dumps({
                'overall_completeness': stats['overall_completeness'],
                'total_patients': len(df.columns),
                'total_fields': len(df.index),
                'available_cells': int(stats['available_cells']),
                'missing_cells': int(stats['missing_cells']),
                'export_date': datetime.now().isoformat()
            }, indent=2)
            
            st.download_button(
                label="Download Statistics",
                data=stats_json,
                file_name=f"cohort_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("Generate Report"):
            st.info("Report generation feature coming soon!")

if __name__ == "__main__":
    main() 





# TODO: join the WSI samples in (this needs to be generic for all sample types here...)