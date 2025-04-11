import io
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Resource manager
        rsrcmgr = PDFResourceManager()
        
        # StringIO to store the extracted text
        output = io.StringIO()
        
        # Create text converter with layout parameters
        device = TextConverter(rsrcmgr, output, laparams=LAParams())
        
        # Create interpreter
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        # Open the PDF
        with open(pdf_path, 'rb') as fp:
            # Process each page
            for page in PDFPage.get_pages(fp, set()):
                interpreter.process_page(page)
        
        # Get text from StringIO
        text = output.getvalue()
        
        # Close resources
        device.close()
        output.close()
        
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def clean_pdf_text(text):
    """
    Clean and normalize extracted PDF text.
    
    Args:
        text (str): Raw text extracted from PDF
        
    Returns:
        str: Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers typically found in PDFs
    text = re.sub(r'\f', '\n', text)  # Form feed characters often separate pages
    
    # Fix common OCR issues
    text = text.replace('|', 'I')  # Pipe character often misrecognized as capital I
    
    # Remove any non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    
    return text.strip()

def extract_sections(text):
    """
    Extract different sections from the document text based on common headers.
    
    Args:
        text (str): Cleaned text from the PDF
        
    Returns:
        dict: Dictionary of section names and their content
    """
    sections = {}
    
    # Define patterns for common section headers in ILD documents
    section_patterns = [
        (r'Brief case summary(.*?)(?=Laboratory|Current medication|$)', 'case_summary'),
        (r'Current medication:(.*?)(?=Laboratory|$)', 'medications'),
        (r'Laboratory(.*?)(?=Pulmonary function test|$)', 'laboratory'),
        (r'Pulmonary function test(.*?)(?=HRCT|$)', 'pulmonary_function'),
        (r'HRCT \[(.*?)\](.*?)(?=Cardiac ultrasound|討論事項及結論|$)', 'hrct'),
        (r'Cardiac ultrasound \[(.*?)\](.*?)(?=討論事項及結論|$)', 'cardiac'),
        (r'討論事項及結論：(.*?)(?=No\.|$)', 'discussion')
    ]
    
    for pattern, section_name in section_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip() if section_name != 'hrct' else {
                'date': match.group(1).strip(),
                'content': match.group(2).strip()
            }
            sections[section_name] = content
    
    return sections

def extract_patient_header(text):
    """
    Extract patient header information containing ID, name, date, etc.
    
    Args:
        text (str): Text from which to extract header
        
    Returns:
        dict: Dictionary with patient header information
    """
    header_info = {}
    
    # Extract patient number, name, ID
    patient_header_pattern = r'No\.\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)'
    patient_match = re.search(patient_header_pattern, text)
    
    if patient_match:
        header_info['number'] = patient_match.group(1).strip()
        header_info['name'] = patient_match.group(2).strip()
        header_info['id'] = patient_match.group(3).strip()
        header_info['date'] = patient_match.group(4).strip()
        header_info['physician'] = patient_match.group(5).strip()
        header_info['diagnosis'] = patient_match.group(6).strip()
    
    # Extract imaging diagnosis (often on the next line)
    imaging_pattern = r'(\S+)\s+imaging_diagnosis:'
    imaging_match = re.search(imaging_pattern, text)
    if imaging_match:
        header_info['imaging_diagnosis'] = imaging_match.group(1).strip()
    
    return header_info
