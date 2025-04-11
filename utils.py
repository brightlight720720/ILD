import uuid
import re
import datetime

def get_session_id():
    """
    Generate a unique session ID.
    
    Returns:
        str: Unique session ID
    """
    return str(uuid.uuid4())

def extract_numeric_value(text):
    """
    Extract numeric value from a string.
    
    Args:
        text (str): Text containing a numeric value
        
    Returns:
        float: Extracted numeric value or None if no value found
    """
    if not text:
        return None
    
    # Remove non-numeric characters except decimal point
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None

def format_date(date_str):
    """
    Format a date string into a standardized format.
    
    Args:
        date_str (str): Date string in various formats
        
    Returns:
        str: Formatted date string or original string if parsing fails
    """
    if not date_str:
        return ""
    
    # Try different date formats
    date_formats = [
        '%Y/%m/%d',  # 2020/04/30
        '%Y-%m-%d',  # 2020-04-30
        '%d/%m/%Y',  # 30/04/2020
        '%m/%d/%Y'   # 04/30/2020
    ]
    
    for fmt in date_formats:
        try:
            date_obj = datetime.datetime.strptime(date_str, fmt)
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return date_str

def clean_text(text):
    """
    Clean text by removing redundant whitespace and normalizing characters.
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def normalize_lab_value(value, test_name):
    """
    Normalize laboratory values for consistent comparison.
    
    Args:
        value (str): The lab value as a string
        test_name (str): The name of the test
        
    Returns:
        float: Normalized value or None if normalization fails
    """
    if not value:
        return None
    
    # Remove qualifiers like '>' or '<'
    value = re.sub(r'[<>]', '', value)
    
    # Remove units and other non-numeric characters
    numeric_match = re.search(r'(\d+\.?\d*)', value)
    if not numeric_match:
        return None
    
    numeric_value = float(numeric_match.group(1))
    
    # Apply test-specific normalization if needed
    if test_name == 'ANA':
        # Convert titers like 1:160 to a numeric scale
        if ':' in value:
            parts = value.split(':')
            if len(parts) == 2 and parts[1].isdigit():
                return float(parts[1])
    
    return numeric_value

def calculate_age(birth_date, reference_date=None):
    """
    Calculate age based on birth date.
    
    Args:
        birth_date (str): Birth date string
        reference_date (str, optional): Reference date for age calculation
        
    Returns:
        int: Age in years or None if calculation fails
    """
    if not birth_date:
        return None
    
    # Format dates
    birth_date = format_date(birth_date)
    
    if reference_date:
        reference_date = format_date(reference_date)
    else:
        reference_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    try:
        birth_date_obj = datetime.datetime.strptime(birth_date, '%Y-%m-%d')
        reference_date_obj = datetime.datetime.strptime(reference_date, '%Y-%m-%d')
        
        age = reference_date_obj.year - birth_date_obj.year
        
        # Adjust age if birthday hasn't occurred yet in the reference year
        if (reference_date_obj.month, reference_date_obj.day) < (birth_date_obj.month, birth_date_obj.day):
            age -= 1
        
        return age
    except ValueError:
        return None
