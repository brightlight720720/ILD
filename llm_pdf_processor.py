import os
import json
import openai
from openai import OpenAI
from pdf_processor import extract_text_from_pdf, clean_pdf_text

# Initialize the OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OpenAI API key not found in environment variables")
    OPENAI_API_KEY = "MISSING_KEY" # This will cause the API to fail properly
openai_client = OpenAI(api_key=OPENAI_API_KEY)
print(f"OpenAI client initialized. API key available: {bool(OPENAI_API_KEY and OPENAI_API_KEY != 'MISSING_KEY')}")

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
            
        # Handle potential text encoding issues
        try:
            # Ensure text is properly encoded
            pdf_text = pdf_text.encode('utf-8', errors='replace').decode('utf-8')
            # Replace any problematic characters
            pdf_text = ''.join(char if ord(char) < 128 else ' ' for char in pdf_text)
        except Exception as e:
            print(f"Warning - text encoding cleanup: {str(e)}")
            # If encoding fails, try a more aggressive cleanup
            pdf_text = ''.join(c for c in pdf_text if c.isprintable())
        
        # Define the extraction prompt
        system_prompt = """
        You are a medical data extraction specialist. Extract structured patient information from the provided text from a PDF about 
        Interstitial Lung Disease (ILD) patient cases. The text is from a multi-disciplinary discussion meeting that contains information
        for MULTIPLE PATIENTS. You MUST identify each separate patient and extract their information individually.

        IMPORTANT: Look for patient separators like "Case X:" or "Discussion X:" or headers that indicate a new patient case.
        The document likely contains 4 separate patient cases.
        
        For each patient, extract these fields (if available):
        1. id: Patient ID number (string)
        2. name: Patient name (string)
        3. case_date: Date of the case (string)
        4. physician: Treating physician (string)
        5. diagnosis: Primary diagnosis (string)
        6. imaging_diagnosis: Imaging diagnosis or impression (string)
        7. case_summary: Brief summary of the patient's condition (string)
        8. medications: Medications as simple key-value pairs where keys are medication categories and values are the specific medications
        9. immunologic_profile: Laboratory immunologic tests as key-value pairs
        10. biologic_markers: Biological markers as key-value pairs
        11. pulmonary_tests: Array of test results with date, FVC, FEV1, etc. as strings
        12. hrct: Object with date, findings, and impression as strings
        13. discussion_points: Array of objects with question and answer fields
        
        Return your response in JSON format using this exact structure:
        {
          "patients": [
            {
              "id": "patient-id-1",
              "name": "Patient Name 1",
              "case_date": "date string",
              ...other fields...
            },
            {
              "id": "patient-id-2",
              "name": "Patient Name 2",
              ...other fields...
            }
          ]
        }
        
        For medications, use a structure like:
        "medications": {
          "Bronchodilator": "medication names",
          "Immunosuppressive agent": "medication names",
          "Anti-fibrotic agent": "medication names"
        }
        
        For pulmonary_tests, use a structure like:
        "pulmonary_tests": [
          {
            "date": "test date",
            "FVC": "value",
            "FEV1": "value",
            "FEV1/FVC": "value",
            "DLCO": "value"
          }
        ]
        
        For discussion_points, use a structure like:
        "discussion_points": [
          {
            "question": "Is this patient's condition considered ILD?",
            "answer": "Yes, based on..."
          }
        ]
        
        If no ID or name is found, use "Patient 1" and "ID-1" for the first patient.
        Use null for any missing fields, not empty strings or objects.
        MAKE SURE to identify and extract ALL patients in the document, likely 4 separate patients.
        """
        
        user_prompt = f"Extract structured patient information from this ILD patient document text and return as JSON. Remember to identify EACH patient separately (likely 4 patients):\n\n{pdf_text}"
        
        # Call the OpenAI API
        try:
            print("Sending request to OpenAI for patient information extraction...")
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
            print("Received response from OpenAI")
        except Exception as api_error:
            print(f"Error calling OpenAI API: {str(api_error)}")
            raise
        
        # Extract and parse the JSON response with robust error handling
        try:
            content = response.choices[0].message.content
            # Clean the content in case there are any invalid characters
            content = content.replace('\n', ' ').replace('\r', ' ')
            print(f"Parsing JSON response. Response length: {len(content)}")
            result = json.loads(content)
            
            # Ensure the result is an array of patients
            patients = result.get("patients", [])
            print(f"Found {len(patients)} patients in the response")
            
            if not isinstance(patients, list):
                # If not a list, try to see if the entire result is the patient array
                if isinstance(result, list):
                    patients = result
                    print("Converted result to patient list")
                else:
                    # Last resort: wrap single patient in a list
                    if isinstance(result, dict) and ("id" in result or "name" in result):
                        patients = [result]
                        print("Converted single patient dict to list")
                    else:
                        patients = []
                        print("No patients found in response")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from OpenAI response: {str(e)}")
            print(f"Response content: {response.choices[0].message.content[:200]}...")
            patients = []
        except Exception as e:
            print(f"Unexpected error processing OpenAI response: {str(e)}")
            patients = []
        
        # Process each patient to ensure required fields
        for i, patient in enumerate(patients):
            print(f"Processing patient {i+1}")
            # Ensure required fields exist
            if "id" not in patient or not patient["id"]:
                patient["id"] = f"ID-{i+1}"
            if "name" not in patient or not patient["name"]:
                patient["name"] = f"Patient {i+1}"
            
            # Ensure discussion_points is properly formatted
            if "discussion_points" in patient and patient["discussion_points"] is None:
                patient["discussion_points"] = []
                
            # Check if pulmonary_tests exists and is properly formatted
            if "pulmonary_tests" in patient and not isinstance(patient["pulmonary_tests"], list):
                if patient["pulmonary_tests"] is None:
                    patient["pulmonary_tests"] = []
                else:
                    # Try to convert non-list to a list with one entry
                    try:
                        patient["pulmonary_tests"] = [patient["pulmonary_tests"]]
                    except:
                        patient["pulmonary_tests"] = []
            
            # Check if medications is properly formatted
            if "medications" in patient and not isinstance(patient["medications"], dict):
                if patient["medications"] is None:
                    patient["medications"] = {}
                else:
                    # Try to convert to a dict if it's not already
                    try:
                        if isinstance(patient["medications"], str):
                            patient["medications"] = {"Other": patient["medications"]}
                    except:
                        patient["medications"] = {}
        
        print(f"Returning {len(patients)} processed patients")
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