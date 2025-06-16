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

def generate_dummy_data(num_patients=50):
    """Generate dummy clinical data for demonstration"""
    
    # Define clinical fields with their characteristic colors
    clinical_fields_colors = {
        'Demographics': '#FF6B6B',      # Red
        'Age': '#FF8E8E', 
        'Gender': '#FFAAAA',
        'BMI': '#FFCCCC',
        'Medical_History': '#4ECDC4',   # Teal
        'Hypertension': '#6DD4CC',
        'Diabetes': '#8ADDD4',
        'Heart_Disease': '#A7E6DD',
        'Lab_Results': '#45B7D1',       # Blue
        'Hemoglobin': '#6BC4DA',
        'White_Blood_Cells': '#8ED1E3',
        'Platelets': '#B1DEEC',
        'Imaging': '#96CEB4',           # Green
        'CT_Scan': '#A8D5BC',
        'MRI': '#BADCC4',
        'X_Ray': '#CCE3CC',
        'Pathology': '#FECA57',         # Yellow/Orange
        'Biopsy_Results': '#FED470',
        'Tumor_Grade': '#FEDD89',
        'Tumor_Stage': '#FEE6A2',
        'Treatment': '#A8E6CF',         # Light Green
        'Surgery': '#B5EAD7',
        'Chemotherapy': '#C2EDDF',
        'Radiation': '#CFF0E7',
        'Follow_up': '#DDA0DD',         # Purple
        'Response': '#E6B3E6',
        'Survival_Status': '#EFC6EF',
        'Last_Visit': '#F8D9F8'
    }
    
    clinical_fields = list(clinical_fields_colors.keys())
    
    # Define 3 sites and distribute patients
    sites = ['Site_A', 'Site_B', 'Site_C']
    patients_per_site = num_patients // 3
    remainder = num_patients % 3
    
    # Generate patient IDs with site information
    patient_ids = []
    site_info = []
    
    for i, site in enumerate(sites):
        site_patients = patients_per_site + (1 if i < remainder else 0)
        for j in range(site_patients):
            patient_id = f"{site}_P{j+1:03d}"
            patient_ids.append(patient_id)
            site_info.append(site)
    
    # Create data availability matrix AND actual data values
    np.random.seed(42)  # For reproducible results
    availability_matrix = []
    actual_data = {}
    
    for field in clinical_fields:
        # Different fields have different availability rates
        if field in ['Demographics', 'Age', 'Gender']:
            availability = 0.95  # Almost always available
        elif field in ['Lab_Results', 'Hemoglobin', 'White_Blood_Cells']:
            availability = 0.85  # Usually available
        elif field in ['Imaging', 'CT_Scan', 'MRI']:
            availability = 0.70  # Sometimes available
        elif field in ['Pathology', 'Biopsy_Results']:
            availability = 0.60  # Less frequently available
        elif field in ['Follow_up', 'Response']:
            availability = 0.40  # Often missing
        else:
            availability = 0.75  # Default availability
        
        # Generate availability for this field
        field_availability = np.random.binomial(1, availability, num_patients)
        availability_matrix.append(field_availability)
        
        # Generate actual data values for available fields
        field_values = []
        for i, patient_id in enumerate(patient_ids):
            if field_availability[i] == 1:  # Data available
                # Generate realistic dummy values based on field type
                if field in ['Age']:
                    field_values.append(np.random.randint(18, 90))
                elif field in ['Gender']:
                    field_values.append(np.random.choice(['Male', 'Female']))
                elif field in ['BMI']:
                    field_values.append(round(np.random.normal(25, 5), 1))
                elif field in ['Hypertension', 'Diabetes', 'Heart_Disease']:
                    field_values.append(np.random.choice(['Yes', 'No']))
                elif field in ['Response']:
                    field_values.append(np.random.choice(['Complete Response', 'Partial Response', 'Stable Disease', 'Progressive Disease']))
                elif field in ['Survival_Status']:
                    field_values.append(np.random.choice(['Alive', 'Deceased']))
                elif field in ['Tumor_Grade']:
                    field_values.append(np.random.choice(['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']))
                elif field in ['Tumor_Stage']:
                    field_values.append(np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV']))
                elif field in ['Hemoglobin']:
                    field_values.append(round(np.random.normal(12.5, 2), 1))
                elif field in ['White_Blood_Cells']:
                    field_values.append(round(np.random.normal(7.5, 2), 1))
                elif field in ['Platelets']:
                    field_values.append(int(np.random.normal(250, 50)))
                else:
                    # Default to binary Yes/No for other fields
                    field_values.append(np.random.choice(['Yes', 'No']))
            else:
                field_values.append(np.nan)  # Missing data
        
        actual_data[field] = field_values
    
    # Convert availability to DataFrame
    availability_df = pd.DataFrame(availability_matrix, index=clinical_fields, columns=patient_ids)
    
    # Convert actual data to DataFrame
    actual_df = pd.DataFrame(actual_data, index=patient_ids).T
    
    return availability_df, actual_df, clinical_fields_colors, site_info

def create_availability_heatmap(df, field_colors, site_info, actual_data_df=None):
    """Create an interactive heatmap showing data availability with characteristic colors"""
    
    # Determine if we're using real cohort data or dummy data
    if len(site_info) == len(df.columns):
        # Real data: group by actual cohorts
        unique_sites = list(set(site_info))
        sites = sorted(unique_sites)  # Sort for consistent ordering
        
        # Group patients by their actual cohort
        site_positions = {}
        current_pos = 0
        
        for site in sites:
            site_patients = [df.columns[i] for i, cohort in enumerate(site_info) if cohort == site]
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
    
    # Create the figure using subplots approach for better control
    fig = make_subplots(rows=1, cols=1)
    
    # Create separate heatmap for each field to get custom colors
    for i, field in enumerate(df.index):
        if field not in field_colors:
            continue
        field_color = field_colors[field]
        
        # Get data for this field (row)
        field_data = df.loc[field].values
        
        # Simple hover info
        hover_info = []
        for j, val in enumerate(field_data):
            patient = df.columns[j]
            status = "Available" if val == 1 else "Missing"
            hover_info.append(f'<b>Field:</b> {field}<br><b>Patient:</b> {patient}<br><b>Status:</b> {status}')
        
        # Create custom colorscale for this field
        colorscale = [[0, '#000000'], [1, field_color]]  # Black to field color
        
        fig.add_trace(go.Heatmap(
            z=[field_data],  # Single row
            x=df.columns,
            y=[field],  # Single field name
            colorscale=colorscale,
            showscale=False,
            hoverongaps=False,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=[hover_info]
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
            tickangle=45,
            side='bottom'
        ),
        yaxis=dict(
            title="Clinical Fields",
            autorange='reversed'  # Keep fields in original order
        ),
        width=1400,
        height=800,
        margin=dict(l=200, r=100, t=100, b=150),
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

def load_real_csv_data(csv_path, subsample_size=None):
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
        
        # Convert to DataFrames
        availability_df = pd.DataFrame(availability_matrix, index=clinical_fields, columns=case_ids)
        actual_df = pd.DataFrame(actual_data, index=case_ids).T
        
        # Create site information from cohorts
        site_info = cohorts.tolist()
        
        return availability_df, actual_df, field_colors, site_info, cohorts.unique()
        
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None, None, None, None

def main():
    # Header
    st.markdown('<div class="main-header">🏥 CohortManager</div>', unsafe_allow_html=True)
    st.markdown("### Clinical Data Availability Dashboard")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Data loading options
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Generate Dummy Data", "Load from Directory", "Load Real CSV Data"],
        help="Choose how to load the clinical data"
    )
    
    if data_source == "Generate Dummy Data":
        num_patients = st.sidebar.slider(
            "Number of Patients",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of patients to generate for demonstration"
        )
        
        # Generate or load cached data
        if 'cohort_data' not in st.session_state or st.sidebar.button("Regenerate Data"):
            with st.spinner("Generating dummy clinical data..."):
                availability_df, actual_df, field_colors, site_info = generate_dummy_data(num_patients)
                st.session_state.cohort_data = availability_df
                st.session_state.actual_data = actual_df
                st.session_state.field_colors = field_colors
                st.session_state.site_info = site_info
        
        df = st.session_state.cohort_data
        actual_data_df = st.session_state.actual_data
        field_colors = st.session_state.field_colors
        site_info = st.session_state.site_info
        
    elif data_source == "Load Real CSV Data":
        csv_path = st.sidebar.text_input(
            "CSV File Path",
            value="/tank/WSI_data/Ovarian_WSIs/OV_master/master_cohort.csv",
            help="Path to the CSV file containing clinical data"
        )
        
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
        
        if csv_path and os.path.exists(csv_path):
            st.sidebar.success("CSV file found!")
            
            # Check if we need to load/reload the data
            if ('cohort_data' not in st.session_state or 
                'csv_path' not in st.session_state or 
                st.session_state.get('csv_path') != csv_path or 
                st.session_state.get('subsample_size') != subsample_size or
                st.sidebar.button("Reload CSV Data")):
                
                with st.spinner("Loading real clinical data..."):
                    # Load real CSV data
                    result = load_real_csv_data(csv_path, subsample_size)
                    
                    if result[0] is not None:  # Check if loading was successful
                        availability_df, actual_df, field_colors, site_info, cohorts = result
                        st.session_state.cohort_data = availability_df
                        st.session_state.actual_data = actual_df
                        st.session_state.field_colors = field_colors
                        st.session_state.site_info = site_info
                        st.session_state.cohorts = cohorts
                        st.session_state.csv_path = csv_path
                        st.session_state.subsample_size = subsample_size
                        sample_text = f" (subsampled from {1031})" if subsample_size else ""
                        st.sidebar.success(f"✅ Loaded {len(availability_df.columns)} patients{sample_text} with {len(availability_df.index)} clinical fields!")
                    else:
                        st.sidebar.error("Failed to load CSV data")
                        # Fallback to dummy data
                        availability_df, actual_df, field_colors, site_info = generate_dummy_data(20)
                        st.session_state.cohort_data = availability_df
                        st.session_state.actual_data = actual_df
                        st.session_state.field_colors = field_colors
                        st.session_state.site_info = site_info
            
            # Use cached data
            df = st.session_state.cohort_data
            actual_data_df = st.session_state.actual_data
            field_colors = st.session_state.field_colors
            site_info = st.session_state.site_info
            
        else:
            st.sidebar.warning("Please provide a valid CSV file path")
            # Use dummy data as fallback
            if 'cohort_data' not in st.session_state:
                availability_df, actual_df, field_colors, site_info = generate_dummy_data(20)
                st.session_state.cohort_data = availability_df
                st.session_state.actual_data = actual_df
                st.session_state.field_colors = field_colors
                st.session_state.site_info = site_info
            
            df = st.session_state.cohort_data
            actual_data_df = st.session_state.actual_data
            field_colors = st.session_state.field_colors
            site_info = st.session_state.site_info
    else:
        data_directory = st.sidebar.text_input(
            "Data Directory Path",
            placeholder="/path/to/clinical/data",
            help="Path to directory containing clinical data files"
        )
        
        if data_directory and os.path.exists(data_directory):
            st.sidebar.success("Directory found!")
            # For now, fall back to dummy data
            # In a real implementation, you would load actual data here
            if 'cohort_data' not in st.session_state:
                availability_df, actual_df, field_colors, site_info = generate_dummy_data(50)
                st.session_state.cohort_data = availability_df
                st.session_state.actual_data = actual_df
                st.session_state.field_colors = field_colors
                st.session_state.site_info = site_info
            df = st.session_state.cohort_data
            actual_data_df = st.session_state.actual_data
            field_colors = st.session_state.field_colors
            site_info = st.session_state.site_info
        else:
            st.sidebar.warning("Please provide a valid directory path")
            availability_df, actual_df, field_colors, site_info = generate_dummy_data(20)  # Small default dataset
            df = availability_df
            actual_data_df = actual_df
    
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
            # Real data: count by actual cohorts
            site_counts = {}
            for cohort in st.session_state.site_info:
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
        
        # Check if we have real data or dummy data
        if data_source == "Load Real CSV Data" and 'field_colors' in st.session_state:
            # Real data: show actual fields
            st.write("**Clinical Fields:**")
            for field in df.index:
                if field in field_colors:
                    st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 5px;"><div style="width: 20px; height: 20px; background-color: {field_colors[field]}; margin-right: 10px; border: 1px solid #ccc;"></div>{field}</div>', unsafe_allow_html=True)
        else:
            # Dummy data: show grouped fields
            color_groups = {
                "Demographics": ["Demographics", "Age", "Gender", "BMI"],
                "Medical History": ["Medical_History", "Hypertension", "Diabetes", "Heart_Disease"],
                "Lab Results": ["Lab_Results", "Hemoglobin", "White_Blood_Cells", "Platelets"],
                "Imaging": ["Imaging", "CT_Scan", "MRI", "X_Ray"],
                "Pathology": ["Pathology", "Biopsy_Results", "Tumor_Grade", "Tumor_Stage"],
                "Treatment": ["Treatment", "Surgery", "Chemotherapy", "Radiation"],
                "Follow-up": ["Follow_up", "Response", "Survival_Status", "Last_Visit"]
            }
            
            for group_name in color_groups:
                with st.expander(f"{group_name} Fields"):
                    for field in color_groups[group_name]:
                        if field in field_colors:
                            st.markdown(f'<div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: {field_colors[field]}; margin-right: 10px; border: 1px solid #ccc;"></div>{field}</div>', unsafe_allow_html=True)
    
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
    st.write("Select clinical fields to create a cohort of patients with complete data for **ALL** selected criteria:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Multi-select for clinical fields
        available_fields = list(df.index)
        selected_fields = st.multiselect(
            "Select Clinical Fields (patients must have data for ALL selected fields):",
            options=available_fields,
            default=[],
            help="Choose multiple fields to create a cohort with complete data for all selected criteria"
        )
        
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
                    # Real data: group by actual cohorts
                    for patient in cohort_patients:
                        patient_idx = list(df.columns).index(patient)
                        cohort = st.session_state.site_info[patient_idx]
                        site_breakdown[cohort] = site_breakdown.get(cohort, 0) + 1
                else:
                    # Dummy data: use original logic
                    for patient in cohort_patients:
                        site = patient.split('_')[0] + '_' + patient.split('_')[1]
                        site_breakdown[site] = site_breakdown.get(site, 0) + 1
                
                st.write("**Cohort by Site/Study:**")
                for site, count in sorted(site_breakdown.items()):
                    st.write(f"• {site}: {count} patients")
                
                # Generate structured filename
                filename_fields = "_".join(selected_fields).replace(" ", "")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                suggested_filename = f"Cohort_{filename_fields}_{timestamp}"
                
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
            
            # Create the export dataframe with actual values
            export_df = actual_data_df[cohort_patients].loc[selected_fields]
            
            # Apply field mappings if any exist
            for field_key, mapping in st.session_state.get('field_mappings', {}).items():
                if mapping:  # Only if mapping is not empty
                    field_name = field_key.split('_')[0]  # Extract field name from key
                    if field_name in export_df.index and len(mapping) > 0:
                        # Apply the mapping
                        export_df.loc[field_name] = export_df.loc[field_name].map(mapping).fillna(export_df.loc[field_name])
            
            with col_csv:
                csv_content = export_df.to_csv()
                st.download_button(
                    label="📄 CSV",
                    data=csv_content,
                    file_name=f"{custom_filename}.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col_json:
                json_content = export_df.to_json(orient='records', indent=2)
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
                    export_df.to_excel(writer, sheet_name='Cohort_Data')
                excel_content = buffer.getvalue()
                
                st.download_button(
                    label="📊 Excel",
                    data=excel_content,
                    file_name=f"{custom_filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Show preview of exported data
            with st.expander("📋 Data Preview", expanded=False):
                st.write("First 5 patients from your cohort:")
                st.dataframe(export_df.iloc[:, :5] if len(export_df.columns) > 5 else export_df)
    
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