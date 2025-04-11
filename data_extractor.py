import re
import pandas as pd
from pdf_processor import clean_pdf_text, extract_sections, extract_patient_header

def extract_patient_data(pdf_text):
    """
    Extract structured patient data from the raw PDF text.
    
    Args:
        pdf_text (str): Raw text extracted from the PDF
        
    Returns:
        list: List of dictionaries containing patient data
    """
    # Clean the PDF text
    cleaned_text = clean_pdf_text(pdf_text)
    
    # Split text into patient sections
    patient_sections = split_by_patients(cleaned_text)
    
    patients_data = []
    for patient_text in patient_sections:
        try:
            patient_data = process_patient_section(patient_text)
            if patient_data:
                patients_data.append(patient_data)
        except Exception as e:
            print(f"Error processing patient section: {str(e)}")
    
    return patients_data

def split_by_patients(text):
    """
    Split the document text into individual patient sections.
    
    Args:
        text (str): Cleaned document text
        
    Returns:
        list: List of text sections, one per patient
    """
    # Pattern to identify the start of a new patient section
    patient_pattern = r'No\.\s+\d+\s+\S+\s+\d+\S+\s+\d+/\d+/\d+'
    
    # Find all matches of patient headers
    matches = list(re.finditer(patient_pattern, text))
    
    # If no matches found, return the entire text as one section
    if not matches:
        return [text]
    
    # Split the text by the matched positions
    patient_sections = []
    for i in range(len(matches)):
        start_pos = matches[i].start()
        end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
        patient_sections.append(text[start_pos:end_pos])
    
    return patient_sections

def process_patient_section(patient_text):
    """
    Process a single patient's text section to extract structured data.
    
    Args:
        patient_text (str): Text section for one patient
        
    Returns:
        dict: Dictionary containing structured patient data
    """
    patient_data = {}
    
    # Extract patient header information
    header_info = extract_patient_header_from_text(patient_text)
    patient_data.update(header_info)
    
    # Extract different sections
    sections = extract_sections(patient_text)
    
    # Process case summary
    if 'case_summary' in sections:
        patient_data['case_summary'] = sections['case_summary']
    
    # Process medications
    if 'medications' in sections:
        patient_data['medications'] = extract_medications(sections['medications'])
    
    # Process laboratory results
    if 'laboratory' in sections:
        immunologic_profile, biologic_markers = extract_lab_results(sections['laboratory'])
        patient_data['immunologic_profile'] = immunologic_profile
        patient_data['biologic_markers'] = biologic_markers
    
    # Process pulmonary function tests
    if 'pulmonary_function' in sections:
        patient_data['pulmonary_tests'] = extract_pulmonary_function_tests(sections['pulmonary_function'])
    
    # Process HRCT findings
    if 'hrct' in sections:
        patient_data['hrct'] = extract_hrct_findings(sections['hrct'])
    
    # Process discussion points
    if 'discussion' in sections:
        patient_data['discussion_points'] = extract_discussion_points(sections['discussion'])
    
    return patient_data

def extract_patient_header_from_text(text):
    """
    Extract patient identification information from the text.
    
    Args:
        text (str): Patient section text
        
    Returns:
        dict: Dictionary containing patient identification data
    """
    patient_info = {}
    
    # Extract patient number
    number_match = re.search(r'No\.\s*(\d+)', text)
    if number_match:
        patient_info['number'] = number_match.group(1).strip()
    
    # Extract patient name
    name_pattern = r'No\.\s*\d+\s+(\S+)'
    name_match = re.search(name_pattern, text)
    if name_match:
        patient_info['name'] = name_match.group(1).strip()
    
    # Extract patient ID
    id_pattern = r'No\.\s*\d+\s+\S+\s+(\d+\S+)'
    id_match = re.search(id_pattern, text)
    if id_match:
        patient_info['id'] = id_match.group(1).strip()
    
    # Extract case date
    date_pattern = r'No\.\s*\d+\s+\S+\s+\d+\S+\s+(\d+/\d+/\d+)'
    date_match = re.search(date_pattern, text)
    if date_match:
        patient_info['case_date'] = date_match.group(1).strip()
    
    # Extract physician
    physician_pattern = r'No\.\s*\d+\s+\S+\s+\d+\S+\s+\d+/\d+/\d+\s+(\S+)'
    physician_match = re.search(physician_pattern, text)
    if physician_match:
        patient_info['physician'] = physician_match.group(1).strip()
    
    # Extract diagnosis
    diagnosis_pattern = r'VS\s+(.+?)(?=影像學診斷|$)'
    diagnosis_match = re.search(diagnosis_pattern, text)
    if diagnosis_match:
        patient_info['diagnosis'] = diagnosis_match.group(1).strip()
    
    # Extract imaging diagnosis
    imaging_pattern = r'影像學診斷\s+(.+?)(?=Brief case|$)'
    imaging_match = re.search(imaging_pattern, text)
    if imaging_match:
        patient_info['imaging_diagnosis'] = imaging_match.group(1).strip()
    
    return patient_info

def extract_medications(medications_text):
    """
    Extract medication information from the text.
    
    Args:
        medications_text (str): Text containing medication information
        
    Returns:
        dict: Dictionary of medication categories and details
    """
    medications = {}
    
    # Common medication categories in ILD documents
    categories = [
        'Bronchodilator', 
        'Immunosuppressive agent', 
        'Anti-fibrotic agent',
        'Pulmonary hypertension agent',
        'Others'
    ]
    
    for category in categories:
        pattern = rf'{category}:\s*(.+?)(?={"|".join(categories)}:|$)'
        match = re.search(pattern, medications_text, re.IGNORECASE | re.DOTALL)
        if match:
            medications[category] = match.group(1).strip()
    
    return medications

def extract_lab_results(lab_text):
    """
    Extract laboratory results from the text.
    
    Args:
        lab_text (str): Text containing laboratory results
        
    Returns:
        tuple: (immunologic_profile, biologic_markers)
    """
    immunologic_profile = {}
    biologic_markers = {}
    
    # Extract immunologic profile items
    immunologic_items = [
        'ANA', 'SS-A', 'SS-B', 'RF', 'Scl-70', 
        'Myositis Ab', 'Jo-1'
    ]
    
    for item in immunologic_items:
        pattern = rf'{item}\s+(.+?)(?=\n|\r|$)'
        match = re.search(pattern, lab_text)
        if match:
            immunologic_profile[item] = match.group(1).strip()
    
    # Extract biologic markers
    biologic_items = [
        'Ferritin', 'ESR', 'hs-CRP', 'CA-199',
        'CA-153', 'CA-125', 'NT-ProBNP', '6MWT'
    ]
    
    for item in biologic_items:
        pattern = rf'{item}\s+(.+?)(?=\n|\r|$)'
        match = re.search(pattern, lab_text)
        if match:
            biologic_markers[item] = match.group(1).strip()
    
    return immunologic_profile, biologic_markers

def extract_pulmonary_function_tests(pft_text):
    """
    Extract pulmonary function test results.
    
    Args:
        pft_text (str): Text containing pulmonary function tests
        
    Returns:
        list: List of dictionaries with test results per date
    """
    # Extract the dates
    date_pattern = r'日期\s+([\d/]+)'
    dates = re.findall(date_pattern, pft_text)
    
    # Define the metrics to extract
    metrics = ['FVC', 'FEV1', 'FEV1/FVC', 'FEF 25-75%', 'TLC', 'DLCO']
    
    # Initialize results list
    results = []
    
    # Extract values for each date
    for i, date in enumerate(dates):
        test_result = {'date': date}
        
        for metric in metrics:
            # Pattern to match the metric and its value
            pattern = rf'{metric}\s+([\d.]+)\s+\(?(\d+%?)?\)?'
            match = re.search(pattern, pft_text)
            
            if match:
                value = match.group(1).strip()
                percentage = match.group(2).strip() if match.group(2) else None
                
                # Store the values
                test_result[metric] = value
                if percentage:
                    test_result[f'{metric}_percent'] = percentage
        
        results.append(test_result)
    
    return results

def extract_hrct_findings(hrct_data):
    """
    Extract HRCT findings from the text.
    
    Args:
        hrct_data (dict): Dict with HRCT date and content
        
    Returns:
        dict: Structured HRCT findings
    """
    hrct_findings = {'date': hrct_data.get('date', '')}
    
    content = hrct_data.get('content', '')
    
    # Extract findings section
    findings_pattern = r'Finding:(.*?)(?=Impression:|$)'
    findings_match = re.search(findings_pattern, content, re.DOTALL)
    if findings_match:
        hrct_findings['findings'] = findings_match.group(1).strip()
    
    # Extract impression section
    impression_pattern = r'Impression:(.*?)$'
    impression_match = re.search(impression_pattern, content, re.DOTALL)
    if impression_match:
        hrct_findings['impression'] = impression_match.group(1).strip()
    
    return hrct_findings

def extract_discussion_points(discussion_text):
    """
    Extract discussion points and answers from the text.
    
    Args:
        discussion_text (str): Text containing discussion points
        
    Returns:
        list: List of dictionaries with questions and answers
    """
    discussion_points = []
    
    # Pattern for numbered questions with yes/no answers
    pattern = r'(\d+)\.\s+(.*?)\s*？\s*(是|否)'
    matches = re.findall(pattern, discussion_text)
    
    for match in matches:
        number, question, answer = match
        discussion_points.append({
            'number': number,
            'question': question.strip(),
            'answer': answer
        })
    
    return discussion_points
