"""
Multi-Agent Medical Team Discussion System using LangChain

This script creates a simulated medical team discussion about a patient with 
interstitial lung disease (ILD), with each agent representing a different medical specialty.
The discussion format is modeled after the ILD multidisciplinary meeting in the provided document.
"""

import os
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

# Set your API key for OpenAI
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Patient case data - You can replace this with data from the document
PATIENT_CASE = """
Patient ID: ILD-2023-045
Age: 58
Gender: Female
Presenting Symptoms: Progressive dyspnea on exertion, dry cough, fatigue
Medical History: 
- SLE overlapping RA for 26 years
- Characterized by arthritis (bilateral knees and PIPs)
- Positive ANA, dsDNA, low C3/C4, antiphospholipid syndrome, high RF, anti-CCP
- Interstitial lung disease was suspected after endoxan pulse therapy in 1995

Physical Examination:
- Bilateral fine crackles at lung bases
- No clubbing or cyanosis
- O2 saturation: 95% on room air

Laboratory Results:
- ANA 1:1280
- SS-A > 240
- SS-B 220
- RF 85.9
- Myositis Ab: Ku:++/ Ro-52:+++
- Ferritin: 9.81
- ESR: 27
- hs-CRP: 0.011
- NT-ProBNP: 78.67

Pulmonary Function Tests:
- FVC: 51% of predicted
- FEV1: 50% of predicted 
- FEV1/FVC: 82%
- DLCO: 40% of predicted

Imaging:
- HRCT: Reticulation over periphery of right lower lobe and lower lobe with honeycombing pattern
- Impression: UIP should be considered, stable compared to previous CT

Current Medications:
- Bronchodilator: Relvar ellipta QD (Fluticasone, VilanterolL ellipta)
- Immunosuppressive agent: AZA 50mg TIW, Tofacitinib 11mg QD, Prednisolone 10mg QD
- Anti-fibrotic agent: none
- Pulmonary hypertension agent: none
- Others: HCQ 200mg QD, montelukast 10mg HS, erythromycin 750mg QD

Cardiac Ultrasound:
- LVH (1.1, 1.1 CM)
- Aortic root dilatation (3.8 CM) with mild AR
- Prolapse of anterior mitral leaflet with mild MR
- LV ejection fraction: 53%
"""

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
    name = "patient_data"
    description = "Tool to access patient data including history, symptoms, lab results, and imaging"
    
    def _run(self, query: str = None) -> str:
        """Return patient data based on the query"""
        return PATIENT_CASE
    
    async def _arun(self, query: str = None) -> str:
        """Return patient data based on the query (async version)"""
        return PATIENT_CASE

class MedicalLiteratureTool(BaseTool):
    name = "medical_literature"
    description = "Tool to access medical literature and guidelines on ILD diagnosis and treatment"
    
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
def setup_multidisciplinary_meeting():
    """Set up the multidisciplinary meeting with all specialists"""
    
    # Create tools
    patient_data_tool = PatientDataTool()
    medical_literature_tool = MedicalLiteratureTool()
    tools = [patient_data_tool, medical_literature_tool]
    
    # Define model to use
    model = "gpt-4" # or gpt-3.5-turbo for lower cost
    
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

# Run the multidisciplinary meeting simulation
def run_multidisciplinary_meeting(specialists, coordinator, tools):
    """Run the multidisciplinary meeting simulation"""
    
    # Meeting setup
    meeting_date = datetime.now().strftime("%Y/%m/%d")
    print(f"ILD Multidisciplinary Team Meeting")
    print(f"Date: {meeting_date}")
    print(f"Location: Virtual Conference")
    print("-" * 80)
    
    # Start with case presentation
    print("CASE PRESENTATION:")
    coordinator_response = coordinator.invoke({
        "input": "Please begin our multidisciplinary ILD meeting by presenting the patient case briefly and inviting initial impressions from the specialists."
    })
    print(f"Coordinator: {coordinator_response['output']}")
    print("-" * 80)
    
    # Initial impressions from specialists
    for specialist_type, specialist in specialists.items():
        input_text = f"As the {specialist_type}, please provide your initial impression of this case based on the available information."
        specialist_response = specialist.invoke({"input": input_text})
        print(f"{specialist_type.title()}: {specialist_response['output']}")
        
        # Add specialist's response to coordinator's memory
        coordinator.memory.chat_memory.add_user_message(f"{specialist_type.title()}: {specialist_response['output']}")
    
    print("-" * 80)
    print("STRUCTURED DISCUSSION:")
    
    # Go through each discussion question
    for question in DISCUSSION_QUESTIONS:
        print(f"\nQuestion: {question}")
        
        # Coordinator asks the question and identifies which specialists should respond
        coordinator_prompt = f"Please ask the team to address the following question: {question}. Identify which specialists should provide their expertise on this question."
        coordinator_response = coordinator.invoke({"input": coordinator_prompt})
        print(f"Coordinator: {coordinator_response['output']}")
        
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
            print(f"{specialist_type.title()}: {specialist_response['output']}")
            
            # Add specialist's response to coordinator's memory
            coordinator.memory.chat_memory.add_user_message(f"{specialist_type.title()}: {specialist_response['output']}")
    
    print("-" * 80)
    print("MEETING CONCLUSION:")
    
    # Coordinator summarizes the discussion and provides final recommendations
    conclusion_prompt = "Please summarize our discussion today and provide the team's consensus on diagnosis and treatment recommendations."
    conclusion_response = coordinator.invoke({"input": conclusion_prompt})
    print(f"Coordinator: {conclusion_response['output']}")

# Main function to run the simulation
def main():
    specialists, coordinator, tools = setup_multidisciplinary_meeting()
    run_multidisciplinary_meeting(specialists, coordinator, tools)

if __name__ == "__main__":
    main()
