import os
from agents import (
    DiagnosisAgent,
    TreatmentAgent,
    ProgressionAgent,
    RiskAssessmentAgent
)

def analyze_patients(patients_data):
    """
    Analyze patient data using multiple specialized agents.
    
    Args:
        patients_data (list): List of patient data dictionaries
        
    Returns:
        list: Analysis results for each patient
    """
    analysis_results = []
    
    # Initialize agents
    diagnosis_agent = DiagnosisAgent()
    treatment_agent = TreatmentAgent()
    progression_agent = ProgressionAgent()
    risk_agent = RiskAssessmentAgent()
    
    for patient in patients_data:
        try:
            # Run analysis with each specialized agent
            diagnosis_analysis = diagnosis_agent.analyze(patient)
            treatment_recommendations = treatment_agent.analyze(patient)
            progression_assessment = progression_agent.analyze(patient)
            risk_assessment = risk_agent.analyze(patient)
            
            # Combine results
            patient_analysis = {
                'patient_id': patient['id'],
                'patient_name': patient['name'],
                'diagnosis_analysis': diagnosis_analysis,
                'treatment_recommendations': treatment_recommendations,
                'progression_assessment': progression_assessment,
                'risk_level': risk_assessment.get('risk_level', 'Unknown'),
                'risk_factors': risk_assessment.get('risk_factors', [])
            }
            
            analysis_results.append(patient_analysis)
        except Exception as e:
            print(f"Error analyzing patient {patient.get('name', 'unknown')}: {str(e)}")
    
    return analysis_results

def categorize_diagnosis(diagnosis, imaging_findings):
    """
    Categorize the diagnosis based on patterns and imaging findings.
    
    Args:
        diagnosis (str): Patient diagnosis
        imaging_findings (str): Imaging diagnosis or findings
        
    Returns:
        dict: Diagnosis categorization
    """
    categorization = {
        'type': 'Unknown',
        'confidence': 'Low',
        'associated_conditions': []
    }
    
    # Check for UIP pattern
    if 'UIP' in imaging_findings:
        categorization['type'] = 'UIP'
        categorization['confidence'] = 'High' if 'definite UIP' in imaging_findings.lower() else 'Moderate'
    
    # Check for NSIP pattern
    elif 'NSIP' in imaging_findings:
        categorization['type'] = 'NSIP'
        categorization['confidence'] = 'High' if 'NSIP pattern' in imaging_findings else 'Moderate'
    
    # Associated conditions
    if 'Sjogren' in diagnosis:
        categorization['associated_conditions'].append('Sjogren\'s syndrome')
    
    if 'SLE' in diagnosis:
        categorization['associated_conditions'].append('Systemic Lupus Erythematosus')
    
    if 'RA' in diagnosis:
        categorization['associated_conditions'].append('Rheumatoid Arthritis')
    
    return categorization

def evaluate_treatment_efficacy(patient_data):
    """
    Evaluate the efficacy of current treatments based on PFT and symptoms.
    
    Args:
        patient_data (dict): Patient data dictionary
        
    Returns:
        dict: Treatment efficacy assessment
    """
    efficacy = {
        'overall': 'Unknown',
        'metrics': {},
        'recommendations': []
    }
    
    # Check PFT trends if available
    if 'pulmonary_tests' in patient_data and len(patient_data['pulmonary_tests']) > 1:
        tests = patient_data['pulmonary_tests']
        
        # Check FVC trend
        if 'FVC' in tests[0] and 'FVC' in tests[-1]:
            fvc_first = float(tests[0]['FVC'])
            fvc_last = float(tests[-1]['FVC'])
            
            if fvc_last < fvc_first * 0.9:
                efficacy['metrics']['FVC'] = 'Declining'
                efficacy['recommendations'].append('Consider intensifying treatment due to declining FVC')
            elif fvc_last > fvc_first * 1.1:
                efficacy['metrics']['FVC'] = 'Improving'
            else:
                efficacy['metrics']['FVC'] = 'Stable'
        
        # Check DLCO trend
        if 'DLCO' in tests[0] and 'DLCO' in tests[-1]:
            dlco_first = float(tests[0]['DLCO'])
            dlco_last = float(tests[-1]['DLCO'])
            
            if dlco_last < dlco_first * 0.85:
                efficacy['metrics']['DLCO'] = 'Declining'
                efficacy['recommendations'].append('Monitor for hypoxemia due to declining DLCO')
            elif dlco_last > dlco_first * 1.15:
                efficacy['metrics']['DLCO'] = 'Improving'
            else:
                efficacy['metrics']['DLCO'] = 'Stable'
    
    # Determine overall efficacy
    if 'metrics' in efficacy and efficacy['metrics']:
        declining_metrics = [m for m, status in efficacy['metrics'].items() if status == 'Declining']
        
        if not declining_metrics:
            efficacy['overall'] = 'Effective'
        elif len(declining_metrics) == len(efficacy['metrics']):
            efficacy['overall'] = 'Ineffective'
        else:
            efficacy['overall'] = 'Partially Effective'
    
    return efficacy

def assess_disease_progression(patient_data):
    """
    Assess the progression of the disease based on patient data.
    
    Args:
        patient_data (dict): Patient data dictionary
        
    Returns:
        dict: Disease progression assessment
    """
    progression = {
        'status': 'Unknown',
        'evidence': [],
        'rate': 'Unknown'
    }
    
    # Check imaging findings for progression
    if 'hrct' in patient_data and 'impression' in patient_data['hrct']:
        impression = patient_data['hrct']['impression'].lower()
        
        if 'stable' in impression:
            progression['status'] = 'Stable'
            progression['evidence'].append('Stable findings on HRCT')
        elif 'progress' in impression or 'worsen' in impression:
            progression['status'] = 'Progressive'
            progression['evidence'].append('Progressive changes on HRCT')
    
    # Check PFT trends
    if 'pulmonary_tests' in patient_data and len(patient_data['pulmonary_tests']) > 1:
        # Calculate rate of decline in FVC if available
        tests = sorted(patient_data['pulmonary_tests'], key=lambda x: x.get('date', ''))
        
        if len(tests) >= 2 and 'FVC_percent' in tests[0] and 'FVC_percent' in tests[-1]:
            try:
                first_fvc = float(tests[0]['FVC_percent'].replace('%', ''))
                last_fvc = float(tests[-1]['FVC_percent'].replace('%', ''))
                
                # Simplified rate calculation (would need dates for proper rate)
                change = last_fvc - first_fvc
                
                if change <= -10:
                    progression['status'] = 'Progressive'
                    progression['rate'] = 'Rapid'
                    progression['evidence'].append(f'Significant FVC decline: {change}%')
                elif change < 0:
                    progression['status'] = 'Progressive'
                    progression['rate'] = 'Slow'
                    progression['evidence'].append(f'Mild FVC decline: {change}%')
                else:
                    progression['status'] = 'Stable'
                    progression['evidence'].append(f'Stable or improved FVC: {change}%')
            except:
                pass
    
    return progression
