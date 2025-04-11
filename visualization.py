import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import re
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

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

def create_risk_assessment_dashboard(patient_data, analysis_data):
    """
    Create a color-coded risk assessment dashboard for ILD patients.
    
    Args:
        patient_data (dict): Patient data dictionary
        analysis_data (dict): Analysis results from the multi-agent system
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object with risk assessment dashboard
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 3, figure=fig)
    
    # Define color scheme for risk levels
    risk_colors = {
        'Low': '#4CAF50',  # Green
        'Moderate': '#FFC107',  # Amber
        'High': '#F44336',  # Red
        'Unknown': '#9E9E9E'  # Gray
    }
    
    # 1. Overall Risk Assessment (Top Row, Spans all columns)
    ax_overall = fig.add_subplot(gs[0, :])
    ax_overall.axis('off')
    
    # Get risk level from analysis data
    risk_level = analysis_data.get('risk_level', 'Unknown')
    risk_factors = analysis_data.get('risk_factors', [])
    
    # Set title and draw colored box for the overall risk
    ax_overall.text(0.5, 0.8, 'Overall Risk Assessment', 
                   fontsize=16, fontweight='bold', ha='center')
    
    # Draw risk level box with appropriate color
    risk_color = risk_colors.get(risk_level, risk_colors['Unknown'])
    rect = mpatches.Rectangle((0.25, 0.2), 0.5, 0.4, 
                             color=risk_color, alpha=0.6, 
                             edgecolor='black', linewidth=1)
    ax_overall.add_patch(rect)
    
    # Add risk level text
    ax_overall.text(0.5, 0.4, f"Risk Level: {risk_level}", 
                   fontsize=14, fontweight='bold', ha='center', va='center')
    
    # 2. Functional Metrics Assessment (Middle Left)
    ax_function = fig.add_subplot(gs[1, 0])
    ax_function.axis('off')
    ax_function.text(0.5, 0.9, 'Pulmonary Function', 
                    fontsize=12, fontweight='bold', ha='center')
    
    # Set defaults
    fvc_percent = 0
    dlco_percent = 0
    
    # Extract PFT data
    if 'pulmonary_tests' in patient_data and patient_data['pulmonary_tests']:
        latest_pft = patient_data['pulmonary_tests'][-1]
        
        # Extract FVC %
        if 'FVC' in latest_pft:
            try:
                fvc_percent = float(str(latest_pft.get('FVC', '')).replace('%', ''))
            except:
                pass
                
        # Extract DLCO %
        if 'DLCO' in latest_pft:
            try:
                dlco_percent = float(str(latest_pft.get('DLCO', '')).replace('%', ''))
            except:
                pass
    
    # Define risk thresholds for PFTs
    # FVC < 50% or DLCO < 35% = High Risk
    # FVC 50-70% or DLCO 35-60% = Moderate Risk
    # FVC > 70% and DLCO > 60% = Low Risk
    
    # Determine PFT risk level
    pft_risk = 'Unknown'
    if fvc_percent > 0 or dlco_percent > 0:
        if fvc_percent < 50 or dlco_percent < 35:
            pft_risk = 'High'
        elif fvc_percent < 70 or dlco_percent < 60:
            pft_risk = 'Moderate'
        else:
            pft_risk = 'Low'
    
    # Draw PFT risk indicator
    rect_pft = mpatches.Rectangle((0.25, 0.4), 0.5, 0.3, 
                                color=risk_colors.get(pft_risk, risk_colors['Unknown']), 
                                alpha=0.6, edgecolor='black', linewidth=1)
    ax_function.add_patch(rect_pft)
    
    # Add PFT metrics text
    ax_function.text(0.5, 0.55, f"FVC: {fvc_percent}%", fontsize=10, ha='center')
    ax_function.text(0.5, 0.45, f"DLCO: {dlco_percent}%", fontsize=10, ha='center')
    
    # 3. Disease Activity (Middle Center)
    ax_activity = fig.add_subplot(gs[1, 1])
    ax_activity.axis('off')
    ax_activity.text(0.5, 0.9, 'Disease Activity', 
                    fontsize=12, fontweight='bold', ha='center')
    
    # Check for rheumatic disease activity markers
    has_activity = False
    if 'immunologic_profile' in patient_data:
        for marker, value in patient_data['immunologic_profile'].items():
            if '+' in str(value) or 'positive' in str(value).lower():
                has_activity = True
                break
                
    if 'biologic_markers' in patient_data:
        if 'ESR' in patient_data['biologic_markers']:
            try:
                esr = float(re.sub(r'[^\d.]', '', patient_data['biologic_markers']['ESR']))
                if esr > 20:  # ESR > 20 indicates disease activity
                    has_activity = True
            except:
                pass
                
        if 'hs-CRP' in patient_data['biologic_markers']:
            try:
                crp = float(re.sub(r'[^\d.]', '', patient_data['biologic_markers']['hs-CRP']))
                if crp > 0.5:  # CRP > 0.5 indicates disease activity
                    has_activity = True
            except:
                pass
    
    # Set activity risk based on findings
    activity_risk = 'Unknown'
    if patient_data.get('diagnosis', '').lower().find('active') >= 0:
        activity_risk = 'High'
    elif has_activity:
        activity_risk = 'Moderate'
    elif 'diagnosis' in patient_data:
        activity_risk = 'Low'
        
    # Draw activity risk indicator
    rect_activity = mpatches.Rectangle((0.25, 0.4), 0.5, 0.3, 
                                     color=risk_colors.get(activity_risk, risk_colors['Unknown']), 
                                     alpha=0.6, edgecolor='black', linewidth=1)
    ax_activity.add_patch(rect_activity)
    
    # Add activity label
    activity_text = "Active" if has_activity else "Stable"
    ax_activity.text(0.5, 0.55, activity_text, fontsize=10, ha='center')
    
    # 4. Treatment Response (Middle Right)
    ax_treatment = fig.add_subplot(gs[1, 2])
    ax_treatment.axis('off')
    ax_treatment.text(0.5, 0.9, 'Treatment Response', 
                     fontsize=12, fontweight='bold', ha='center')
    
    # Determine treatment response based on analysis data
    treatment_response = 'Unknown'
    treatment_text = ""
    
    if 'treatment_recommendations' in analysis_data:
        treatment_text = analysis_data['treatment_recommendations']
        # Check treatment response from analysis
        if 'poor response' in treatment_text.lower() or 'inadequate' in treatment_text.lower():
            treatment_response = 'High'
        elif 'partial response' in treatment_text.lower() or 'moderate' in treatment_text.lower():
            treatment_response = 'Moderate'
        elif 'good response' in treatment_text.lower() or 'adequate' in treatment_text.lower():
            treatment_response = 'Low'
    
    # Draw treatment response indicator
    rect_treatment = mpatches.Rectangle((0.25, 0.4), 0.5, 0.3, 
                                      color=risk_colors.get(treatment_response, risk_colors['Unknown']), 
                                      alpha=0.6, edgecolor='black', linewidth=1)
    ax_treatment.add_patch(rect_treatment)
    
    # Add treatment response label
    response_text = "Responsive" if treatment_response == 'Low' else "Non-responsive" if treatment_response == 'High' else "Partial"
    ax_treatment.text(0.5, 0.55, response_text, fontsize=10, ha='center')
    
    # 5. Progression Indicators (Bottom Row, Spans all columns)
    ax_progression = fig.add_subplot(gs[2, :])
    ax_progression.axis('off')
    ax_progression.text(0.5, 0.9, 'Disease Progression Indicators', 
                       fontsize=12, fontweight='bold', ha='center')
    
    # Get progression assessment from analysis data
    progression_text = analysis_data.get('progression_assessment', '')
    
    # Determine progression risk
    progression_risk = 'Unknown'
    if progression_text:
        if 'rapid' in progression_text.lower() or 'significant' in progression_text.lower():
            progression_risk = 'High'
        elif 'moderate' in progression_text.lower() or 'slow' in progression_text.lower():
            progression_risk = 'Moderate'
        elif 'stable' in progression_text.lower() or 'minimal' in progression_text.lower():
            progression_risk = 'Low'
    
    # Draw progression risk indicator boxes
    rect_progression = mpatches.Rectangle((0.1, 0.2), 0.8, 0.4, 
                                        color=risk_colors.get(progression_risk, risk_colors['Unknown']), 
                                        alpha=0.3, edgecolor='black', linewidth=1)
    ax_progression.add_patch(rect_progression)
    
    # Add risk factors text
    if risk_factors:
        factor_text = "Risk Factors: " + ", ".join(risk_factors[:3])
        if len(risk_factors) > 3:
            factor_text += f" (+{len(risk_factors) - 3} more)"
        ax_progression.text(0.5, 0.4, factor_text, fontsize=10, ha='center')
    
    # Add legend at the bottom
    handles = [mpatches.Patch(color=color, label=level) for level, color in risk_colors.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    return fig
