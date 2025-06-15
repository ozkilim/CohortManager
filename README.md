# CohortManager

🏥 **CohortManager** is a Streamlit-based web application for visualizing and managing clinical cohort data. It provides an intuitive interface to analyze data availability across patients and clinical fields through interactive heatmaps and statistical summaries.

## Features

- **Interactive Data Availability Matrix**: Visualize which clinical data is available for each patient using color-coded heatmaps
- **Statistical Analysis**: Get comprehensive statistics about data completeness at patient and field levels
- **Flexible Data Loading**: Support for dummy data generation or loading from directories
- **Multiple Visualization Options**: Choose from different color schemes and display options
- **Export Capabilities**: Export data and statistics in CSV and JSON formats
- **Responsive Design**: Modern, mobile-friendly interface with customizable layouts

## Screenshots

The app displays:
- **Main Dashboard**: Overview metrics showing total patients, clinical fields, and overall completeness
- **Data Availability Matrix**: Interactive heatmap where green cells indicate available data and red cells indicate missing data
- **Statistical Charts**: Histograms and bar charts showing completeness distributions
- **Export Options**: Download processed data and statistics

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ozkilim/CohortManager.git
   cd CohortManager
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Alternative Installation (Virtual Environment)

```bash
# Create virtual environment
python -m venv cohort_env

# Activate virtual environment
# On Windows:
cohort_env\Scripts\activate
# On macOS/Linux:
source cohort_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Usage

### Getting Started

1. **Launch the app** using `streamlit run app.py`
2. **Choose your data source** from the sidebar:
   - **Generate Dummy Data**: Creates realistic clinical data for demonstration
   - **Load from Directory**: Load actual clinical data from a specified directory
3. **Adjust parameters** like number of patients (for dummy data)
4. **Explore the visualization** using the interactive heatmap
5. **Analyze statistics** using the detailed charts and metrics

### Understanding the Visualization

- **Columns**: Represent individual patients (P_001, P_002, etc.)
- **Rows**: Represent clinical data fields (Demographics, Lab Results, Imaging, etc.)
- **Colors**: 
  - 🟢 **Green**: Data is available
  - 🔴 **Red**: Data is missing
  - **Color intensity** can be adjusted using different color schemes

### Data Fields

The application tracks various clinical data categories:

- **Demographics**: Age, Gender, BMI, Smoking History
- **Medical History**: Hypertension, Diabetes, Heart Disease
- **Laboratory Results**: Hemoglobin, White Blood Cells, Platelets
- **Imaging**: CT Scan, MRI, X-Ray, Ultrasound
- **Pathology**: Biopsy Results, Tumor Grade, Tumor Stage
- **Treatment**: Surgery, Chemotherapy, Radiation
- **Follow-up**: Response, Survival Status, Last Visit

## Configuration

### Sidebar Options

- **Data Source**: Choose between dummy data generation or directory loading
- **Number of Patients**: Adjust the cohort size (10-200 patients)
- **Display Options**: Toggle patient and field statistics
- **Color Schemes**: Select from Green/Red, Blue/Gray, or Purple/Orange

### Customization

The app can be customized by modifying:
- **Clinical fields** in the `generate_dummy_data()` function
- **Data availability rates** for different field types
- **Color schemes** in the heatmap creation function
- **Export formats** and statistics calculations

## File Structure

```
CohortManager/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── data/              # Directory for clinical data files (optional)
```

## Development

### Adding New Features

1. **New Data Sources**: Extend the data loading functionality in the sidebar section
2. **Additional Visualizations**: Add new chart types in the statistics section
3. **Export Formats**: Implement additional export options (Excel, PDF reports)
4. **Data Validation**: Add data quality checks and validation rules

### Code Structure

- `generate_dummy_data()`: Creates synthetic clinical data for testing
- `create_availability_heatmap()`: Generates the main visualization
- `calculate_statistics()`: Computes completeness metrics
- `main()`: Main application flow and UI layout

## Deployment

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Cloud Deployment

**Streamlit Cloud**:
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Deploy with one click

**Docker**:
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on GitHub or contact the development team.

## Changelog

### v1.0.0 (Current)
- Initial release
- Interactive data availability heatmap
- Statistical analysis and visualization
- Export functionality
- Responsive design

## Roadmap

- [ ] Support for real clinical data formats (CSV, JSON, XML)
- [ ] Advanced filtering and search capabilities
- [ ] Multi-cohort comparison views
- [ ] Automated data quality reports
- [ ] Integration with clinical databases
- [ ] User authentication and access control 