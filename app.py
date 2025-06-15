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
    
    # Create data availability matrix (1 = data available, 0 = missing)
    np.random.seed(42)  # For reproducible results
    data_matrix = []
    
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
        field_data = np.random.binomial(1, availability, num_patients)
        data_matrix.append(field_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data_matrix, index=clinical_fields, columns=patient_ids)
    
    return df, clinical_fields_colors, site_info

def create_availability_heatmap(df, field_colors, site_info):
    """Create an interactive heatmap showing data availability with characteristic colors"""
    
    # Create custom colorscale data for each row
    fig = go.Figure()
    
    # Group patients by site for visual separation
    sites = ['Site_A', 'Site_B', 'Site_C']
    site_positions = {}
    current_pos = 0
    
    for site in sites:
        site_patients = [col for col in df.columns if col.startswith(site)]
        if site_patients:
            site_positions[site] = (current_pos, current_pos + len(site_patients) - 1)
            current_pos += len(site_patients)
    
    # Create the main heatmap with custom colors
    z_values = []
    colors = []
    hover_text = []
    
    for i, field in enumerate(df.index):
        field_color = field_colors[field]
        row_colors = []
        row_hover = []
        
        for j, patient in enumerate(df.columns):
            if df.iloc[i, j] == 1:  # Data available
                row_colors.append(field_color)
                status = "Available"
            else:  # Data missing
                row_colors.append('#000000')  # Black for missing
                status = "Missing"
            
            row_hover.append(f"Patient: {patient}<br>Field: {field}<br>Status: {status}")
        
        colors.append(row_colors)
        hover_text.append(row_hover)
    
    # Create individual traces for each row to have different colors
    for i, field in enumerate(df.index):
        # Convert row data to show field-specific colors
        row_data = []
        for j, patient in enumerate(df.columns):
            if df.iloc[i, j] == 1:  # Available - use field color
                row_data.append(1)
            else:  # Missing - will be black
                row_data.append(0)
        
        fig.add_trace(go.Heatmap(
            z=[row_data],
            x=df.columns,
            y=[field],
            colorscale=[[0, '#000000'], [1, field_colors[field]]],  # Black to field color
            showscale=False,
            hoverongaps=False,
            hovertemplate=f'<b>Patient:</b> %{{x}}<br><b>Field:</b> {field}<br><b>Status:</b> %{{customdata}}<extra></extra>',
            customdata=[["Available" if val == 1 else "Missing" for val in row_data]]
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
                line=dict(color="white", width=3)
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
            tickmode='linear',
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

def main():
    # Header
    st.markdown('<div class="main-header">🏥 CohortManager</div>', unsafe_allow_html=True)
    st.markdown("### Clinical Data Availability Dashboard")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Data loading options
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Generate Dummy Data", "Load from Directory"],
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
                df, field_colors, site_info = generate_dummy_data(num_patients)
                st.session_state.cohort_data = df
                st.session_state.field_colors = field_colors
                st.session_state.site_info = site_info
        
        df = st.session_state.cohort_data
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
                df, field_colors, site_info = generate_dummy_data(50)
                st.session_state.cohort_data = df
                st.session_state.field_colors = field_colors
                st.session_state.site_info = site_info
            df = st.session_state.cohort_data
            field_colors = st.session_state.field_colors
            site_info = st.session_state.site_info
        else:
            st.sidebar.warning("Please provide a valid directory path")
            df, field_colors, site_info = generate_dummy_data(20)  # Small default dataset
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Main content area
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        # Count patients per site
        site_counts = {}
        for col in df.columns:
            site = col.split('_')[0] + '_' + col.split('_')[1]
            site_counts[site] = site_counts.get(site, 0) + 1
        
        st.metric(
            label="Sites",
            value=len(site_counts),
            delta=f"A:{site_counts.get('Site_A', 0)}, B:{site_counts.get('Site_B', 0)}, C:{site_counts.get('Site_C', 0)}"
        )
    
    with col4:
        st.metric(
            label="Overall Completeness",
            value=f"{stats['overall_completeness']:.1f}%"
        )
    
    with col5:
        st.metric(
            label="Available Data Points",
            value=f"{stats['available_cells']:,} / {stats['total_cells']:,}"
        )
    
    # Main visualization
    st.subheader("Data Availability Matrix")
    
    # Display options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Display Options")
        show_patient_stats = st.checkbox("Show Patient Statistics", value=True)
        show_field_stats = st.checkbox("Show Field Statistics", value=True)
        
        # Show color legend
        st.subheader("Field Colors")
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
        fig = create_availability_heatmap(df, field_colors, site_info)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional statistics
    if show_patient_stats or show_field_stats:
        st.subheader("Detailed Statistics")
        
        col1, col2 = st.columns(2)
        
        if show_patient_stats:
            with col1:
                st.subheader("Patient Completeness")
                
                # Create patient completeness chart
                patient_df = pd.DataFrame({
                    'Patient': stats['patient_completeness'].index,
                    'Completeness': stats['patient_completeness'].values
                })
                
                fig_patients = px.histogram(
                    patient_df,
                    x='Completeness',
                    nbins=20,
                    title='Distribution of Patient Data Completeness',
                    labels={'Completeness': 'Completeness (%)', 'count': 'Number of Patients'}
                )
                
                st.plotly_chart(fig_patients, use_container_width=True)
                
                # Show patients with lowest completeness
                worst_patients = stats['patient_completeness'].nsmallest(5)
                st.write("**Patients with Lowest Completeness:**")
                for patient, completeness in worst_patients.items():
                    st.write(f"• {patient}: {completeness:.1f}%")
        
        if show_field_stats:
            with col2:
                st.subheader("Field Completeness")
                
                # Create field completeness chart
                field_df = pd.DataFrame({
                    'Field': stats['field_completeness'].index,
                    'Completeness': stats['field_completeness'].values
                })
                
                fig_fields = px.bar(
                    field_df.sort_values('Completeness', ascending=True),
                    x='Completeness',
                    y='Field',
                    orientation='h',
                    title='Field Data Completeness',
                    labels={'Completeness': 'Completeness (%)', 'Field': 'Clinical Field'}
                )
                
                fig_fields.update_layout(height=600)
                st.plotly_chart(fig_fields, use_container_width=True)
    
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