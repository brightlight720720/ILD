"""
Multi-Agent Medical Team Discussion System using LangChain

This module creates a simulated medical team discussion about a patient with 
interstitial lung disease (ILD), with each agent representing a different medical specialty.
The discussion format is modeled after the ILD multidisciplinary meeting.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.agent import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool

# Get the OpenAI API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OpenAI API key not found in environment variables for LangChain agents")
    OPENAI_API_KEY = "MISSING_KEY"  # This will cause API calls to fail properly
print(f"LangChain Agents - OpenAI API key available: {bool(OPENAI_API_KEY and OPENAI_API_KEY != 'MISSING_KEY')}")

# Define discussion questions based on the document format
DISCUSSION_QUESTIONS = [
    "Is this patient's condition considered ILD?",
    "Is this an indeterminate case?",
    "Does the patient have UIP pattern?",
    "Is there any NSIP pattern present?",
    "Is there ongoing rheumatic disease activity?",
    "Is the ILD progressing?",
    "Should we adjust the immunosuppressive therapy?",
    "Should we recommend anti-fibrotic medication?"
]

# Create tools for accessing medical data
class PatientDataTool(BaseTool):
    name: str = "patient_data"
    description: str = "Tool to access patient data including history, symptoms, lab results, and imaging"
    patient_data_str: str = ""  # Add field to store the data as a string
    
    def __init__(self, patient_data):
        super().__init__()
        self.patient_data_str = json.dumps(patient_data, ensure_ascii=False, indent=2)
    
    def _run(self, query: str = None) -> str:
        """Return patient data based on the query"""
        return self.patient_data_str
    
    async def _arun(self, query: str = None) -> str:
        """Return patient data based on the query (async version)"""
        return self.patient_data_str

class MedicalLiteratureTool(BaseTool):
    name: str = "medical_literature"
    description: str = "Tool to access medical literature and guidelines on ILD diagnosis and treatment"
    
    def _run(self, query: str) -> str:
        """Return relevant medical literature about ILD based on query"""
        literature = {
            "uip": "Usual Interstitial Pneumonia (UIP) is characterized by patchy involvement of the lung parenchyma, with areas of fibrosis and honeycombing alternating with areas of less affected or normal parenchyma. Key radiographic features include reticular opacities, predominantly basal and peripheral distribution, honeycombing, and traction bronchiectasis with minimal ground-glass opacities.",
            "nsip": "Nonspecific Interstitial Pneumonia (NSIP) is characterized by varying degrees of inflammation and fibrosis, with a more uniform appearance than UIP. Ground-glass opacities are more prominent, and honeycombing is typically absent or minimal. Bilateral, symmetric involvement with lower lung predominance is common.",
            "ctd-ild": "Connective Tissue Disease-associated ILD (CTD-ILD) occurs in patients with autoimmune conditions like rheumatoid arthritis, systemic sclerosis, and Sjögren's syndrome. Treatment typically involves immunosuppression with consideration of anti-fibrotic agents in progressive cases.",
            "treatment": "Treatment approaches for ILD vary based on the underlying cause. For CTD-ILD, immunosuppressive therapy is the mainstay. For IPF with UIP pattern, anti-fibrotic medications like nintedanib or pirfenidone are recommended. In cases with mixed patterns or progression despite immunosuppression, combination therapy may be considered.",
            "progression": "ILD progression is typically monitored through pulmonary function tests (particularly FVC and DLCO), symptom assessment, and serial imaging. A decline in FVC >10% or DLCO >15% within 6-12 months is considered clinically significant progression."
        }
        
        for key, value in literature.items():
            if key in query.lower():
                return value
        
        return "No specific information found for this query in the medical literature database."
    
    async def _arun(self, query: str) -> str:
        """Return relevant medical literature about ILD based on query (async version)"""
        return self._run(query)

# Define specialized LLM agents for each medical specialty
def create_specialist(model, specialist_type, specialist_description, tools, temperature=0.2):
    """Create a specialist agent with a specific medical specialty"""
    
    # Define the system message that establishes the agent's specialty and role
    system_message = f"""You are an experienced {specialist_type} physician specialist. 
{specialist_description}
    
You are participating in a multidisciplinary team meeting discussing a patient with 
suspected interstitial lung disease. Consider the patient information carefully, and 
provide your perspective based on your specialty.

Respond in a professional medical tone, using appropriate medical terminology while 
being concise and focused on your area of expertise. When discussing with other specialists,
acknowledge their input and build upon the dialogue constructively.

Remember that this is a collaborative discussion with the goal of developing the 
best care plan for the patient."""

    # Create the prompt template for the specialist
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")]
    )
    
    # Create the LLM for the specialist with appropriate temperature
    llm = ChatOpenAI(temperature=temperature, model=model)
    
    # Create the specialist agent
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    
    # Create memory for the agent
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the agent executor
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

# Create the coordinator to facilitate the meeting
def create_coordinator(model, tools):
    """Create a coordinator agent to facilitate the multidisciplinary meeting"""
    
    system_message = """You are the coordinator of a multidisciplinary ILD meeting.
Your role is to:
1. Facilitate discussion between specialists
2. Ask focused questions about the case
3. Summarize key points
4. Guide the team through the standard discussion questions
5. Ensure each specialist contributes their expertise
6. Maintain a professional, collaborative environment
7. Build consensus on diagnosis and treatment recommendations

Begin by introducing the case and inviting initial impressions from each specialist.
Then proceed through the standard discussion questions, ensuring each relevant 
specialist provides input. Finally, work toward a consensus on diagnosis and 
treatment plan."""

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")]
    )
    
    llm = ChatOpenAI(temperature=0.2, model=model)
    
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

# Setup the simulation
def setup_multidisciplinary_meeting(patient_data):
    """Set up the multidisciplinary meeting with all specialists"""
    
    # Create tools
    patient_data_tool = PatientDataTool(patient_data)
    medical_literature_tool = MedicalLiteratureTool()
    tools = [patient_data_tool, medical_literature_tool]
    
    # Define model to use - Using gpt-4o which is the newest model
    model = "gpt-4o"
    
    # Define specialists
    specialists = {
        "pulmonologist": create_specialist(
            model, 
            "Pulmonologist", 
            """You specialize in respiratory medicine with expertise in interstitial lung diseases.
            Your focus is on lung function, imaging patterns, and distinguishing between different 
            ILD subtypes. You have extensive experience with HRCT interpretation for ILD and 
            understand the nuances of UIP vs NSIP patterns.""",
            tools
        ),
        
        "rheumatologist": create_specialist(
            model, 
            "Rheumatologist", 
            """You specialize in rheumatic diseases with particular expertise in connective 
            tissue disease-associated ILD. You focus on the autoimmune aspects of the case,
            immunological profiles, and immunosuppressive treatment approaches.""",
            tools
        ),
        
        "radiologist": create_specialist(
            model, 
            "Thoracic Radiologist", 
            """You specialize in thoracic imaging with expertise in HRCT evaluation of ILD.
            You can distinguish between different ILD patterns including UIP and NSIP based
            on imaging characteristics. You provide detailed analysis of distribution patterns,
            honeycombing, ground glass opacities, and other relevant imaging findings.""",
            tools
        ),
        
        "pathologist": create_specialist(
            model, 
            "Pathologist", 
            """You specialize in lung pathology with expertise in ILD diagnosis. You interpret
            biopsy findings and correlate them with clinical and radiographic data. You understand
            the histopathological features of UIP, NSIP, and other ILD patterns.""",
            tools
        ),
        
        "cardiologist": create_specialist(
            model, 
            "Cardiologist", 
            """You specialize in cardiac complications of pulmonary disease with a focus on
            pulmonary hypertension. You interpret echocardiogram findings and advise on
            cardiac implications of ILD.""",
            tools
        )
    }
    
    # Create coordinator
    coordinator = create_coordinator(model, tools)
    
    return specialists, coordinator, tools

def run_multidisciplinary_meeting(patient_data):
    """Run the multidisciplinary meeting simulation and return structured results"""
    specialists, coordinator, tools = setup_multidisciplinary_meeting(patient_data)
    
    results = {
        "meeting_date": datetime.now().strftime("%Y/%m/%d"),
        "patient_id": patient_data.get("id", "Unknown"),
        "patient_name": patient_data.get("name", "Unknown"),
        "case_presentation": "",
        "specialist_impressions": {},
        "discussion_points": {},
        "conclusion": "",
        "diagnosis_analysis": "",
        "treatment_recommendations": "",
        "progression_assessment": "",
        "risk_assessment": {
            "risk_level": "Unknown",
            "risk_factors": []
        }
    }
    
    # Start with case presentation
    coordinator_response = coordinator.invoke({
        "input": "Please begin our multidisciplinary ILD meeting by presenting the patient case briefly and inviting initial impressions from the specialists."
    })
    results["case_presentation"] = coordinator_response['output']
    
    # Initial impressions from specialists
    for specialist_type, specialist in specialists.items():
        input_text = f"As the {specialist_type}, please provide your initial impression of this case based on the available information."
        specialist_response = specialist.invoke({"input": input_text})
        results["specialist_impressions"][specialist_type] = specialist_response['output']
        
        # Add specialist's response to coordinator's memory
        coordinator.memory.chat_memory.add_user_message(f"{specialist_type.title()}: {specialist_response['output']}")
    
    # Go through each discussion question
    for question in DISCUSSION_QUESTIONS:
        # Coordinator asks the question and identifies which specialists should respond
        coordinator_prompt = f"Please ask the team to address the following question: {question}. Identify which specialists should provide their expertise on this question."
        coordinator_response = coordinator.invoke({"input": coordinator_prompt})
        
        discussion_point = {
            "coordinator_prompt": coordinator_response['output'],
            "specialist_responses": {}
        }
        
        # Extract mentioned specialists from coordinator's response
        response_lower = coordinator_response['output'].lower()
        responding_specialists = []
        for specialist_type in specialists.keys():
            if specialist_type.lower() in response_lower:
                responding_specialists.append(specialist_type)
        
        # If no specialists were explicitly mentioned, include all
        if not responding_specialists:
            responding_specialists = list(specialists.keys())
        
        # Get responses from the identified specialists
        for specialist_type in responding_specialists:
            specialist = specialists[specialist_type]
            context = f"The coordinator has asked the team to address: {question}"
            specialist_response = specialist.invoke({"input": context})
            discussion_point["specialist_responses"][specialist_type] = specialist_response['output']
            
            # Add specialist's response to coordinator's memory
            coordinator.memory.chat_memory.add_user_message(f"{specialist_type.title()}: {specialist_response['output']}")
        
        results["discussion_points"][question] = discussion_point
    
    # Coordinator summarizes the discussion and provides final recommendations
    conclusion_prompt = "Please summarize our discussion today and provide the team's consensus on diagnosis and treatment recommendations."
    conclusion_response = coordinator.invoke({"input": conclusion_prompt})
    results["conclusion"] = conclusion_response['output']
    
    # Extract specific analyses
    diagnosis_prompt = "Based on our discussion, please provide a concise diagnosis analysis for this patient."
    diagnosis_response = coordinator.invoke({"input": diagnosis_prompt})
    results["diagnosis_analysis"] = diagnosis_response['output']
    
    treatment_prompt = "Based on our discussion, please provide specific treatment recommendations for this patient."
    treatment_response = coordinator.invoke({"input": treatment_prompt})
    results["treatment_recommendations"] = treatment_response['output']
    
    progression_prompt = "Based on our discussion, please provide an assessment of the disease progression for this patient."
    progression_response = coordinator.invoke({"input": progression_prompt})
    results["progression_assessment"] = progression_response['output']
    
    risk_prompt = "Based on our discussion, please provide a risk assessment for this patient, including risk level (low, moderate, high) and specific risk factors."
    risk_response = coordinator.invoke({"input": risk_prompt})
    
    # Try to extract structured risk assessment
    try:
        risk_text = risk_response['output'].lower()
        if "high risk" in risk_text or "severe risk" in risk_text:
            risk_level = "High"
        elif "moderate risk" in risk_text or "medium risk" in risk_text:
            risk_level = "Moderate"
        elif "low risk" in risk_text or "mild risk" in risk_text:
            risk_level = "Low"
        else:
            risk_level = "Unknown"
            
        results["risk_assessment"]["risk_level"] = risk_level
        
        # Extract risk factors (simplified approach)
        risk_factors = []
        lines = risk_response['output'].split('\n')
        for line in lines:
            if line.strip().startswith('-') or line.strip().startswith('*'):
                risk_factors.append(line.strip().lstrip('-*').strip())
        
        if risk_factors:
            results["risk_assessment"]["risk_factors"] = risk_factors
        else:
            # Fallback if no bullet points found
            results["risk_assessment"]["risk_factors"] = ["Risk factors not clearly identified"]
            
    except Exception as e:
        print(f"Error extracting structured risk assessment: {e}")
        results["risk_assessment"]["explanation"] = risk_response['output']
    
    return results

def analyze_patient_with_langchain(patient_data):
    """
    Analyze a single patient using the LangChain multi-agent system.
    
    Args:
        patient_data (dict): Patient data dictionary
        
    Returns:
        dict: Analysis results
    """
    try:
        # Run the multidisciplinary meeting simulation
        results = run_multidisciplinary_meeting(patient_data)
        
        # Format the results for compatibility with the existing app
        analysis_results = {
            'patient_id': patient_data['id'],
            'patient_name': patient_data['name'],
            'diagnosis_analysis': results['diagnosis_analysis'],
            'treatment_recommendations': results['treatment_recommendations'],
            'progression_assessment': results['progression_assessment'],
            'risk_level': results['risk_assessment']['risk_level'],
            'risk_factors': results['risk_assessment']['risk_factors'],
            'specialist_impressions': results['specialist_impressions'],
            'meeting_discussion': results['discussion_points'],
            'meeting_conclusion': results['conclusion']
        }
        
        return analysis_results
    except Exception as e:
        print(f"Error analyzing patient {patient_data.get('name', 'unknown')}: {str(e)}")
        return {
            'patient_id': patient_data.get('id', 'unknown'),
            'patient_name': patient_data.get('name', 'unknown'),
            'diagnosis_analysis': f"Analysis could not be completed: {str(e)}",
            'treatment_recommendations': "No recommendations available due to analysis error",
            'progression_assessment': "No assessment available due to analysis error",
            'risk_level': "Unknown",
            'risk_factors': ["Analysis error"]
        }

def analyze_patients_with_langchain(patients_data):
    """
    Analyze multiple patients using the LangChain multi-agent system.
    
    Args:
        patients_data (list): List of patient data dictionaries
        
    Returns:
        list: Analysis results for each patient
    """
    analysis_results = []
    
    for patient in patients_data:
        try:
            patient_analysis = analyze_patient_with_langchain(patient)
            analysis_results.append(patient_analysis)
        except Exception as e:
            print(f"Error analyzing patient {patient.get('name', 'unknown')}: {str(e)}")
    
    return analysis_results