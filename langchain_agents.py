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
from llm_providers import llm_manager
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool

# Import RAG knowledge base
try:
    from rag_knowledge_base import ILDKnowledgeBase, enhance_tool_with_rag
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("RAG functionality not available - using standard medical literature tool")

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
def create_specialist(specialist_type, specialist_description, tools, temperature=0.2):
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
    llm = llm_manager.get_chat_model(temperature=temperature)
    
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
def create_coordinator(tools):
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
    
    llm = llm_manager.get_chat_model(temperature=0.2)
    
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
    
    # Set up RAG knowledge base if available
    if RAG_AVAILABLE:
        try:
            print("Initializing RAG Knowledge Base for ILD medical literature...")
            knowledge_base = ILDKnowledgeBase(use_openai=True)
            # Enhance the medical literature tool with RAG capabilities
            medical_literature_tool = enhance_tool_with_rag(medical_literature_tool, knowledge_base)
            print("Successfully initialized RAG-enhanced medical literature tool")
        except Exception as e:
            print(f"Error initializing RAG Knowledge Base: {e}")
            print("Falling back to standard medical literature tool")
    
    tools = [patient_data_tool, medical_literature_tool]
    
    # Define specialists
    specialists = {
        "pulmonologist": create_specialist(
            "Pulmonologist", 
            """You specialize in respiratory medicine with expertise in interstitial lung diseases.
            Your focus is on lung function, imaging patterns, and distinguishing between different 
            ILD subtypes. You have extensive experience with HRCT interpretation for ILD and 
            understand the nuances of UIP vs NSIP patterns.""",
            tools
        ),
        
        "rheumatologist": create_specialist(
            "Rheumatologist", 
            """You specialize in rheumatic diseases with particular expertise in connective 
            tissue disease-associated ILD. You focus on the autoimmune aspects of the case,
            immunological profiles, and immunosuppressive treatment approaches.""",
            tools
        ),
        
        "radiologist": create_specialist(
            "Thoracic Radiologist", 
            """You specialize in thoracic imaging with expertise in HRCT evaluation of ILD.
            You can distinguish between different ILD patterns including UIP and NSIP based
            on imaging characteristics. You provide detailed analysis of distribution patterns,
            honeycombing, ground glass opacities, and other relevant imaging findings.""",
            tools
        ),
        
        # Pathologist removed as requested
        
        "cardiologist": create_specialist(
            "Cardiologist", 
            """You specialize in cardiac complications of pulmonary disease with a focus on
            pulmonary hypertension. You interpret echocardiogram findings and advise on
            cardiac implications of ILD.""",
            tools
        )
    }
    
    # Create coordinator
    coordinator = create_coordinator(tools)
    
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
    
    # Initial impressions from specialists - made more concise
    for specialist_type, specialist in specialists.items():
        input_text = f"""As the {specialist_type}, please provide your VERY BRIEF initial impression of this case.
        Limit your response to 50 words maximum. Focus only on the most critical findings from your specialty's perspective."""
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
        
        # Get responses from the identified specialists - made more concise
        for specialist_type in responding_specialists:
            specialist = specialists[specialist_type]
            context = f"""The coordinator has asked the team to address: {question}
            As the {specialist_type}, please provide a VERY BRIEF response limited to 60 words maximum.
            Focus only on the most critical points from your specialty's perspective."""
            specialist_response = specialist.invoke({"input": context})
            discussion_point["specialist_responses"][specialist_type] = specialist_response['output']
            
            # Add specialist's response to coordinator's memory
            coordinator.memory.chat_memory.add_user_message(f"{specialist_type.title()}: {specialist_response['output']}")
        
        results["discussion_points"][question] = discussion_point
    
    # Coordinator summarizes the discussion and provides final recommendations
    conclusion_prompt = """Please provide an EXTREMELY CONCISE summary of our discussion today (maximum 100 words).
    Focus only on the most critical findings and consensus recommendations.
    Be direct and straight to the point with minimal explanations."""
    conclusion_response = coordinator.invoke({"input": conclusion_prompt})
    results["conclusion"] = conclusion_response['output']
    
    # Have the coordinator answer the 8 key clinical questions in Chinese with simple yes/no answers
    key_questions = {
        "是否為 ILD": "Is this patient's condition considered ILD?",
        "是否為 Indeterminate": "Is this an indeterminate case?",
        "是否為 UIP": "Does the patient have UIP pattern?",
        "是否還有 NSIP pattern": "Is there any NSIP pattern present?", 
        "是否還有免風疾病活動性(activity) 病變": "Is there ongoing rheumatic disease activity?",
        "是否 ILD 持續進展": "Is the ILD progressing?",
        "是否調整免疫治療藥物": "Should we adjust the immunosuppressive therapy?",
        "是否建議使用抗肺纖維化藥物": "Should we recommend anti-fibrotic medication?"
    }
    
    # Create a specific prompt for the coordinator to answer all 8 questions with clear yes/no
    questions_prompt = """
    Based on our team's discussion, please provide clear YES (是) or NO (否) answers to each of the following 8 key clinical questions:
    
    1. 是否為 ILD? (Is this patient's condition considered ILD?)
    2. 是否為 Indeterminate? (Is this an indeterminate case?)
    3. 是否為 UIP? (Does the patient have UIP pattern?)
    4. 是否還有 NSIP pattern? (Is there any NSIP pattern present?)
    5. 是否還有免風疾病活動性(activity) 病變? (Is there ongoing rheumatic disease activity?)
    6. 是否 ILD 持續進展? (Is the ILD progressing?)
    7. 是否調整免疫治療藥物? (Should we adjust the immunosuppressive therapy?)
    8. 是否建議使用抗肺纖維化藥物? (Should we recommend anti-fibrotic medication?)
    
    For each question, provide ONLY the answer "是" (YES) or "否" (NO), formatted exactly as:
    是否為 ILD: 是
    """
    
    questions_response = coordinator.invoke({"input": questions_prompt})
    
    # Parse the response to extract the yes/no answers
    specific_questions = {}
    response_text = questions_response['output']
    
    # Extract answers for each question
    for zh_question, en_question in key_questions.items():
        # Try with the Chinese question first
        for line in response_text.split('\n'):
            if zh_question in line:
                answer = "是" if "是" in line.split(":", 1)[1].strip() else "否"
                specific_questions[zh_question] = answer
                break
        
        # If not found, try with simplified patterns
        if zh_question not in specific_questions:
            # Extract keywords for each question
            keywords = {
                "是否為 ILD": ["ILD", "interstitial lung disease"],
                "是否為 Indeterminate": ["indeterminate", "uncertain", "unclear"],
                "是否為 UIP": ["UIP", "usual interstitial pneumonia"],
                "是否還有 NSIP pattern": ["NSIP", "non-specific interstitial pneumonia"],
                "是否還有免風疾病活動性(activity) 病變": ["rheumatic", "autoimmune", "activity"],
                "是否 ILD 持續進展": ["progress", "progression", "progressing", "worsen"],
                "是否調整免疫治療藥物": ["adjust", "immunosuppressive", "change therapy"],
                "是否建議使用抗肺纖維化藥物": ["anti-fibrotic", "antifibrotic", "pirfenidone", "nintedanib"]
            }
            
            # Set default answer based on context clues in the discussion
            for question_num, (question, line_num) in enumerate(zip(DISCUSSION_QUESTIONS, range(1, 9))):
                line_prefix = f"{line_num}."
                for line in response_text.split('\n'):
                    if line_prefix in line and any(kw.lower() in line.lower() for kw in keywords[zh_question]):
                        answer = "是" if "yes" in line.lower() or "是" in line else "否"
                        specific_questions[zh_question] = answer
                        break
            
            # If still not found, use default based on DISCUSSION_QUESTIONS answers
            if zh_question not in specific_questions:
                for i, q in enumerate(DISCUSSION_QUESTIONS):
                    if any(kw in q.lower() for kw in [k.lower() for k in keywords[zh_question]]):
                        for line in response_text.split('\n'):
                            if f"{i+1}." in line:
                                answer = "是" if "yes" in line.lower() or "是" in line else "否"
                                specific_questions[zh_question] = answer
                                break
            
            # Final fallback if still not found
            if zh_question not in specific_questions:
                specific_questions[zh_question] = "否"  # Default to "No"
    
    results["specific_questions"] = specific_questions
    
    # Extract specific analyses with more concise output
    diagnosis_prompt = """Based on our discussion, please provide a VERY CONCISE diagnosis analysis for this patient.
    Limit your response to 100 words maximum. Focus only on key diagnostic findings and conclusions."""
    diagnosis_response = coordinator.invoke({"input": diagnosis_prompt})
    results["diagnosis_analysis"] = diagnosis_response['output']
    
    treatment_prompt = """Based on our discussion, please provide EXTREMELY CONCISE treatment recommendations for this patient.
    Limit your response to 80 words maximum. List only specific medications and interventions with minimal explanation."""
    treatment_response = coordinator.invoke({"input": treatment_prompt})
    results["treatment_recommendations"] = treatment_response['output']
    
    progression_prompt = """Based on our discussion, please provide a VERY BRIEF assessment of the disease progression for this patient.
    Limit your response to 75 words maximum. Simply state if disease is stable, progressing, or improving, with only key evidence."""
    progression_response = coordinator.invoke({"input": progression_prompt})
    results["progression_assessment"] = progression_response['output']
    
    risk_prompt = """Based on our discussion, please provide a VERY BRIEF risk assessment for this patient.
    State the risk level (low, moderate, high) and list ONLY 3-4 most critical risk factors.
    Limit your response to 60 words maximum and use bullet points for clarity."""
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
            'meeting_conclusion': results['conclusion'],
            'specific_questions': results['specific_questions']  # Include the directly answered key clinical questions
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
            'risk_factors': ["Analysis error"],
            'specific_questions': {}  # Empty specific questions for error case
        }

def analyze_patients_with_langchain(patients_data, use_rag=True):
    """
    Analyze multiple patients using the LangChain multi-agent system.
    
    Args:
        patients_data (list): List of patient data dictionaries
        use_rag (bool, optional): Whether to use RAG for enhancing medical knowledge. Defaults to True.
        
    Returns:
        list: Analysis results for each patient
    """
    # Set global flag for RAG usage
    global RAG_AVAILABLE
    # Store original flag to restore later
    original_rag_flag = RAG_AVAILABLE if not use_rag else None
    
    try:
        analysis_results = []
        
        # Print status about RAG usage
        if use_rag and RAG_AVAILABLE:
            print("Using RAG-enhanced medical knowledge for patient analysis")
        elif not use_rag:
            print("RAG medical knowledge enhancement disabled by user")
            # Temporarily disable RAG
            RAG_AVAILABLE = False
        else:
            print("RAG not available, using standard medical literature")
        
        for patient in patients_data:
            try:
                patient_analysis = analyze_patient_with_langchain(patient)
                analysis_results.append(patient_analysis)
            except Exception as e:
                print(f"Error analyzing patient {patient.get('name', 'unknown')}: {str(e)}")
                # Add a basic error result
                error_result = {
                    'patient_id': patient.get('id', 'unknown'),
                    'patient_name': patient.get('name', 'unknown'),
                    'diagnosis_analysis': f"Analysis could not be completed: {str(e)}",
                    'treatment_recommendations': "No recommendations available due to analysis error",
                    'progression_assessment': "No assessment available due to analysis error",
                    'risk_level': "Unknown",
                    'risk_factors': ["Analysis error"],
                    'specific_questions': {}
                }
                analysis_results.append(error_result)
        
        return analysis_results
    finally:
        # Restore the original RAG flag if it was temporarily disabled
        if original_rag_flag is not None:
            RAG_AVAILABLE = original_rag_flag