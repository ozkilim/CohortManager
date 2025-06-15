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
    
    # Define clinical fields
    clinical_fields = [
        'Demographics', 'Age', 'Gender', 'BMI', 'Smoking_History',
        'Medical_History', 'Hypertension', 'Diabetes', 'Heart_Disease',
        'Lab_Results', 'Hemoglobin', 'White_Blood_Cells', 'Platelets',
        'Imaging', 'CT_Scan', 'MRI', 'X_Ray', 'Ultrasound',
        'Pathology', 'Biopsy_Results', 'Tumor_Grade', 'Tumor_Stage',
        'Treatment', 'Surgery', 'Chemotherapy', 'Radiation',
        'Follow_up', 'Response', 'Survival_Status', 'Last_Visit'
    ]
    
    # Generate patient IDs
    patient_ids = [f"P_{i:03d}" for i in range(1, num_patients + 1)]
    
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
    
    return df

def create_availability_heatmap(df):
    """Create an interactive heatmap showing data availability"""
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale=[[0, '#ffcccc'], [1, '#2ecc71']],  # Light red for missing, green for available
        showscale=True,
        colorbar=dict(
            title="Data Availability",
            tickvals=[0, 1],
            ticktext=["Missing", "Available"],
            len=0.5
        ),
        hoverongaps=False,
        hovertemplate='<b>Patient:</b> %{x}<br><b>Field:</b> %{y}<br><b>Status:</b> %{customdata}<extra></extra>',
        customdata=[["Available" if val == 1 else "Missing" for val in row] for row in df.values]
    ))
    
    fig.update_layout(
        title={
            'text': 'Clinical Data Availability Matrix',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(
            title="Patients",
            tickangle=45,
            side='bottom'
        ),
        yaxis=dict(
            title="Clinical Fields",
            tickmode='linear'
        ),
        width=1200,
        height=800,
        margin=dict(l=200, r=100, t=100, b=150)
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
                st.session_state.cohort_data = generate_dummy_data(num_patients)
        
        df = st.session_state.cohort_data
        
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
                st.session_state.cohort_data = generate_dummy_data(50)
            df = st.session_state.cohort_data
        else:
            st.sidebar.warning("Please provide a valid directory path")
            df = generate_dummy_data(20)  # Small default dataset
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
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
        st.metric(
            label="Overall Completeness",
            value=f"{stats['overall_completeness']:.1f}%"
        )
    
    with col4:
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
        
        # Color scheme selection
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Green/Red", "Blue/Gray", "Purple/Orange"],
            help="Choose color scheme for the heatmap"
        )
    
    with col1:
        # Create and display the heatmap
        fig = create_availability_heatmap(df)
        
        # Update color scheme if needed
        if color_scheme == "Blue/Gray":
            fig.data[0].colorscale = [[0, '#d3d3d3'], [1, '#1f77b4']]
        elif color_scheme == "Purple/Orange":
            fig.data[0].colorscale = [[0, '#ffd700'], [1, '#9932cc']]
        
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