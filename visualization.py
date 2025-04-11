import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io

def plot_pulmonary_function_trends(pft_df):
    """
    Create a visualization of pulmonary function test trends.
    
    Args:
        pft_df (pandas.DataFrame): DataFrame containing PFT results
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check which metrics we have percentage values for
    metrics_to_plot = []
    for metric in ['FVC', 'FEV1', 'DLCO']:
        percent_col = f'{metric}_percent'
        if percent_col in pft_df.columns:
            metrics_to_plot.append(metric)
    
    # If no percentage values, use absolute values
    if not metrics_to_plot:
        metrics_to_plot = ['FVC', 'FEV1', 'DLCO']
        use_percent = False
    else:
        use_percent = True
    
    # Colors for each metric
    colors = {'FVC': 'blue', 'FEV1': 'green', 'DLCO': 'red'}
    
    # Extract dates and convert to datetime if possible
    dates = pft_df['date'].tolist()
    
    # Plot each metric
    for metric in metrics_to_plot:
        if metric in pft_df.columns:
            if use_percent:
                values = [float(str(val).replace('%', '')) for val in pft_df[f'{metric}_percent'] if pd.notnull(val)]
                label = f"{metric} (%)"
            else:
                values = [float(val) for val in pft_df[metric] if pd.notnull(val)]
                label = metric
            
            if len(values) == len(dates):
                ax.plot(dates, values, 'o-', color=colors.get(metric, 'black'), label=label)
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Value (%)')
    ax.set_title('Pulmonary Function Test Trends')
    ax.grid(True)
    ax.legend()
    
    # Rotate x-axis labels if they're dates
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

def create_lab_results_radar(immunologic_profile, biologic_markers):
    """
    Create a radar chart for laboratory results visualization.
    
    Args:
        immunologic_profile (dict): Immunologic profile results
        biologic_markers (dict): Biologic marker results
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object
    """
    # This is a simplified implementation since radar charts 
    # require numerical values and normalization
    
    # Select metrics that can be normalized
    metrics = {}
    
    # Try to extract numerical values from immunologic profile
    for key, value in immunologic_profile.items():
        try:
            # Extract numbers from strings like "85.9" or "> 240"
            num_value = float(re.sub(r'[^\d.]', '', value))
            metrics[key] = num_value
        except:
            continue
    
    # Similarly for biologic markers
    for key, value in biologic_markers.items():
        try:
            num_value = float(re.sub(r'[^\d.]', '', value))
            metrics[key] = num_value
        except:
            continue
    
    if not metrics:
        # No valid numerical data for radar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Insufficient numerical data for radar chart', 
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig
    
    # Create radar chart
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw the chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    plt.xticks(angles[:-1], list(metrics.keys()))
    
    # Plot data
    values = list(metrics.values())
    values += values[:1]  # Close the loop
    
    # Normalize the values for display
    max_val = max(values)
    normalized_values = [v / max_val for v in values]
    
    ax.plot(angles, normalized_values, linewidth=1, linestyle='solid')
    ax.fill(angles, normalized_values, alpha=0.1)
    
    # Add legend or title
    plt.title('Laboratory Results Overview (Normalized)')
    
    return fig

def create_patient_summary_table(patient_data):
    """
    Create a summary table for a patient's key metrics.
    
    Args:
        patient_data (dict): Patient data dictionary
        
    Returns:
        pandas.DataFrame: Summary DataFrame
    """
    summary_data = {}
    
    # Add basic patient info
    summary_data['Patient Name'] = patient_data.get('name', 'N/A')
    summary_data['Patient ID'] = patient_data.get('id', 'N/A')
    summary_data['Case Date'] = patient_data.get('case_date', 'N/A')
    summary_data['Diagnosis'] = patient_data.get('diagnosis', 'N/A')
    
    # Add latest PFT values if available
    if 'pulmonary_tests' in patient_data and patient_data['pulmonary_tests']:
        latest_pft = patient_data['pulmonary_tests'][-1]
        for metric in ['FVC', 'FEV1', 'DLCO']:
            if metric in latest_pft:
                percent_key = f'{metric}_percent'
                percent = latest_pft.get(percent_key, 'N/A')
                summary_data[f'Latest {metric}'] = f"{latest_pft[metric]} ({percent})"
    
    # Latest lab values
    if 'biologic_markers' in patient_data:
        for key in ['ESR', 'hs-CRP', 'Ferritin']:
            if key in patient_data['biologic_markers']:
                summary_data[key] = patient_data['biologic_markers'][key]
    
    # Convert to DataFrame for tabular display
    df = pd.DataFrame([summary_data])
    return df
