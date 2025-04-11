import os
import json
import openai
from openai import OpenAI
from pdf_processor import extract_text_from_pdf, clean_pdf_text

# Initialize the OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def extract_patient_info_with_llm(pdf_text):
    """
    Use OpenAI's model to extract structured patient information from PDF text.
    
    Args:
        pdf_text (str): Raw or cleaned text from the PDF
        
    Returns:
        list: List of dictionaries containing structured patient data
    """
    try:
        # Clean text if needed
        if len(pdf_text) > 200000:  # If text is very long, truncate it
            pdf_text = pdf_text[:200000]
        
        # Define the extraction prompt
        system_prompt = """
        You are a medical data extraction specialist. Extract structured patient information from the provided text from a PDF about 
        Interstitial Lung Disease (ILD) patient cases. The text is from a multi-disciplinary discussion meeting.
        
        Extract ALL information for EACH patient in the document. Some patients might have partial information, extract whatever is available.
        
        For each patient, extract the following fields (if available):
        1. ID: Patient ID number
        2. name: Patient name 
        3. case_date: Date of the case
        4. physician: Treating physician
        5. diagnosis: Primary diagnosis
        6. imaging_diagnosis: Imaging diagnosis or impression
        7. case_summary: Brief summary of the patient's condition
        8. medications: All medications organized by category
        9. immunologic_profile: Laboratory immunologic tests and results
        10. biologic_markers: Biological markers and values
        11. pulmonary_tests: Pulmonary function test results with dates
        12. hrct: HRCT findings with date and impressions
        13. discussion_points: Points discussed in the meeting and conclusions
        
        Return the data as a structured JSON array where each patient is an object.
        Each patient MUST have at least id and name fields. If these cannot be extracted, use "Patient X" and "ID-X" where X is the order in the document.
        If a field has no information, use null or an empty structure appropriate for that field.
        """
        
        user_prompt = f"Extract structured patient information from this ILD patient document text:\n\n{pdf_text}"
        
        # Call the OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Use the latest model for best results
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,  # Lower temperature for more deterministic output
            max_tokens=4000
        )
        
        # Extract and parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Ensure the result is an array of patients
        patients = result.get("patients", [])
        if not isinstance(patients, list):
            # If not a list, try to see if the entire result is the patient array
            if isinstance(result, list):
                patients = result
            else:
                # Last resort: wrap single patient in a list
                if isinstance(result, dict) and ("id" in result or "name" in result):
                    patients = [result]
                else:
                    patients = []
        
        # Process each patient to ensure required fields
        for i, patient in enumerate(patients):
            # Ensure required fields exist
            if "id" not in patient or not patient["id"]:
                patient["id"] = f"ID-{i+1}"
            if "name" not in patient or not patient["name"]:
                patient["name"] = f"Patient {i+1}"
        
        return patients
    
    except Exception as e:
        print(f"Error extracting patient information with LLM: {str(e)}")
        # Return an empty list in case of error
        return []

def process_pdf_with_llm(pdf_path):
    """
    Process a PDF document using LLM for information extraction.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of dictionaries containing patient data
    """
    try:
        # Extract text from PDF
        raw_text = extract_text_from_pdf(pdf_path)
        
        # Clean the text
        cleaned_text = clean_pdf_text(raw_text)
        
        # Use LLM to extract patient information
        patients_data = extract_patient_info_with_llm(cleaned_text)
        
        return patients_data
    
    except Exception as e:
        print(f"Error processing PDF with LLM: {str(e)}")
        return []