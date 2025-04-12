import json
import os
from openai import OpenAI
import re

# Get the OpenAI API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    OPENAI_API_KEY = "MISSING_KEY"  # This will cause API calls to fail properly but not crash

# Initialize OpenAI client
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
client = OpenAI(api_key=OPENAI_API_KEY)
print(f"Agents - OpenAI API key available: {bool(OPENAI_API_KEY and OPENAI_API_KEY != 'MISSING_KEY')}")

class BaseAgent:
    """Base class for all specialized agents."""
    
    def __init__(self):
        self.model = "gpt-4o"
    
    def analyze(self, patient_data):
        """
        Analyze patient data and return insights.
        
        Args:
            patient_data (dict): Patient data dictionary
            
        Returns:
            str or dict: Analysis results
        """
        # Convert patient data to a string representation
        patient_json = json.dumps(patient_data, ensure_ascii=False, indent=2)
        
        # Define the prompt
        prompt = self._create_prompt(patient_json)
        
        try:
            # Make the API call
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract and process the response
            analysis = response.choices[0].message.content
            
            return self._process_response(analysis)
        except Exception as e:
            print(f"Error in agent analysis: {str(e)}")
            return "Analysis could not be completed due to an error."
    
    def _create_prompt(self, patient_json):
        """Create a prompt for the agent. Override in subclasses."""
        return f"Analyze the following patient data:\n{patient_json}"
    
    def _get_system_prompt(self):
        """Get the system prompt for the agent. Override in subclasses."""
        return "You are a medical assistant specializing in interstitial lung disease (ILD)."
    
    def _process_response(self, response):
        """Process the response from the API. Override in subclasses if needed."""
        return response


class DiagnosisAgent(BaseAgent):
    """Agent specializing in diagnosis verification and assessment."""
    
    def _get_system_prompt(self):
        return """You are a pulmonologist specializing in interstitial lung disease (ILD) diagnosis.
        Your task is to analyze patient data and provide insights on the diagnosis.
        Focus on:
        1. Verifying the ILD diagnosis based on HRCT patterns and clinical findings
        2. Assessing if the pattern is UIP, NSIP, or other
        3. Evaluating the confidence level of the diagnosis
        4. Identifying any alternative diagnostic considerations
        
        Be extremely concise and to the point. Limit your total response to 500 words or less.
        Use clear, direct medical language without unnecessary explanations.
        """
    
    def _create_prompt(self, patient_json):
        return f"""Analyze the following ILD patient data and provide a diagnosis assessment.
        Pay special attention to:
        - The HRCT findings and patterns (UIP vs NSIP vs other)
        - Laboratory results that support or contradict the diagnosis
        - Pulmonary function test results and their implications
        - The relationship between the rheumatologic diagnosis and the lung findings
        
        Patient Data:
        {patient_json}
        
        Provide a VERY CONCISE diagnostic assessment (under 500 words total).
        Clearly answer YES or NO to these specific questions at the end of your analysis:
        1. 是否為 ILD？是 否
        2. 是否為 Indeterminate？是 否
        3. 是否為 UIP？是 否
        4. 是否還有 NSIP pattern？是 否
        5. 是否還有免風疾病活動性(activity) 病變？是 否
        6. 是否 ILD 持續進展？是 否
        7. 是否調整免疫治療藥物？是 否
        8. 是否建議使用抗肺纖維化藥物？是 否
        """


class TreatmentAgent(BaseAgent):
    """Agent specializing in treatment recommendations."""
    
    def _get_system_prompt(self):
        return """You are a specialist in ILD treatment. Your task is to analyze patient data and 
        provide treatment recommendations based on current evidence and guidelines.
        Focus on:
        1. Assessing the appropriateness of current medications
        2. Suggesting potential modifications to the treatment plan
        3. Recommending additional therapies if indicated
        4. Considering both anti-inflammatory and anti-fibrotic approaches
        
        Be extremely concise and to the point. Limit your total response to 500 words or less.
        Use clear, direct medical language without unnecessary explanations.
        """
    
    def _create_prompt(self, patient_json):
        return f"""Review the following ILD patient data and provide treatment recommendations.
        Consider:
        - The current medication regimen and its appropriateness
        - Whether immunosuppressive therapy should be intensified, maintained, or reduced
        - Whether anti-fibrotic therapy is indicated
        - Any supportive therapies that should be considered
        - Management of comorbidities
        
        Patient Data:
        {patient_json}
        
        Provide VERY CONCISE, actionable treatment recommendations (under 500 words total).
        Clearly answer YES or NO to these specific questions at the end of your analysis:
        1. 是否為 ILD？是 否
        2. 是否為 Indeterminate？是 否
        3. 是否為 UIP？是 否
        4. 是否還有 NSIP pattern？是 否
        5. 是否還有免風疾病活動性(activity) 病變？是 否
        6. 是否 ILD 持續進展？是 否
        7. 是否調整免疫治療藥物？是 否
        8. 是否建議使用抗肺纖維化藥物？是 否
        """


class ProgressionAgent(BaseAgent):
    """Agent specializing in disease progression assessment."""
    
    def _get_system_prompt(self):
        return """You are a specialist in monitoring ILD progression. Your task is to analyze 
        patient data for evidence of disease stability or progression over time.
        Focus on:
        1. Identifying trends in pulmonary function tests
        2. Assessing changes in HRCT findings
        3. Evaluating symptom progression
        4. Determining the rate of progression if present
        
        Be extremely concise and to the point. Limit your total response to 500 words or less.
        Use clear, direct medical language without unnecessary explanations.
        """
    
    def _create_prompt(self, patient_json):
        return f"""Analyze the following ILD patient data for evidence of disease progression.
        Specifically assess:
        - Trends in FVC, FEV1, and DLCO values
        - Comparative HRCT findings over time
        - Changes in oxygenation or exercise tolerance
        - Development of new symptoms or complications
        
        Patient Data:
        {patient_json}
        
        Provide a VERY CONCISE assessment of disease stability or progression (under 500 words total).
        Clearly answer YES or NO to these specific questions at the end of your analysis:
        1. 是否為 ILD？是 否
        2. 是否為 Indeterminate？是 否
        3. 是否為 UIP？是 否
        4. 是否還有 NSIP pattern？是 否
        5. 是否還有免風疾病活動性(activity) 病變？是 否
        6. 是否 ILD 持續進展？是 否
        7. 是否調整免疫治療藥物？是 否
        8. 是否建議使用抗肺纖維化藥物？是 否
        """


class RiskAssessmentAgent(BaseAgent):
    """Agent specializing in risk assessment."""
    
    def _get_system_prompt(self):
        return """You are a specialist in ILD risk assessment. Your task is to analyze patient data
        and determine risk levels for adverse outcomes.
        Focus on:
        1. Identifying risk factors for disease progression
        2. Assessing mortality risk
        3. Evaluating risk for complications like pulmonary hypertension
        4. Determining risks associated with therapy
        
        Be extremely concise and to the point. Limit your total explanation to 500 words or less.
        Use clear, direct medical language without unnecessary explanations.
        """
    
    def _create_prompt(self, patient_json):
        return f"""Analyze the following ILD patient data and provide a comprehensive risk assessment.
        Consider:
        - Risk factors for rapid disease progression
        - Indicators of increased mortality risk
        - Risk for development of pulmonary hypertension
        - Side effect risks from current or potential therapies
        - Comorbidities that may complicate management
        
        Patient Data:
        {patient_json}
        
        Provide a VERY CONCISE risk assessment with risk level and specific risk factors (explanation under 500 words).
        FORMAT YOUR RESPONSE AS JSON with the following structure:
        {{
            "risk_level": "low|moderate|high",
            "risk_factors": ["factor1", "factor2", ...],
            "explanation": "concise explanation under 500 words",
            "specific_questions": {{
                "是否為 ILD": "是|否",
                "是否為 Indeterminate": "是|否",
                "是否為 UIP": "是|否",
                "是否還有 NSIP pattern": "是|否",
                "是否還有免風疾病活動性(activity) 病變": "是|否",
                "是否 ILD 持續進展": "是|否",
                "是否調整免疫治療藥物": "是|否",
                "是否建議使用抗肺纖維化藥物": "是|否"
            }}
        }}
        
        Make sure to respond with valid JSON only, and fill in each of the 8 specific questions in Chinese.
        """
    
    def _process_response(self, response):
        """Process the response to extract structured risk assessment."""
        try:
            # Try to parse as JSON first
            result = json.loads(response)
            # Ensure specific_questions exists
            if "specific_questions" not in result:
                result["specific_questions"] = {
                    "是否為 ILD": "否",
                    "是否為 Indeterminate": "否",
                    "是否為 UIP": "否",
                    "是否還有 NSIP pattern": "否",
                    "是否還有免風疾病活動性(activity) 病變": "否",
                    "是否 ILD 持續進展": "否",
                    "是否調整免疫治療藥物": "否",
                    "是否建議使用抗肺纖維化藥物": "否"
                }
            return result
        except json.JSONDecodeError:
            # If not valid JSON, extract information using regex
            risk_level_match = re.search(r"risk_level[\"']?\s*:\s*[\"']?([a-zA-Z]+)[\"']?", response)
            risk_level = risk_level_match.group(1) if risk_level_match else "Unknown"
            
            risk_factors = []
            risk_factors_match = re.search(r"risk_factors[\"']?\s*:\s*\[(.*?)\]", response, re.DOTALL)
            if risk_factors_match:
                factors_text = risk_factors_match.group(1)
                factors = re.findall(r"[\"']([^\"']+)[\"']", factors_text)
                risk_factors = factors
            
            # Try to extract specific questions
            specific_questions = {}
            questions = [
                "是否為 ILD",
                "是否為 Indeterminate",
                "是否為 UIP",
                "是否還有 NSIP pattern",
                "是否還有免風疾病活動性(activity) 病變",
                "是否 ILD 持續進展",
                "是否調整免疫治療藥物",
                "是否建議使用抗肺纖維化藥物"
            ]
            
            for question in questions:
                pattern = f"{question}[：:]\s*[是否]"
                match = re.search(pattern, response)
                if match:
                    answer = "是" if "是" in match.group(0) else "否"
                    specific_questions[question] = answer
                else:
                    specific_questions[question] = "否"  # Default to no if not found
            
            return {
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "explanation": response,
                "specific_questions": specific_questions
            }
