import streamlit as st
import os
import tempfile
import pandas as pd
import json
import matplotlib.pyplot as plt
import io
import re

from pdf_processor import extract_text_from_pdf
from llm_pdf_processor import process_pdf_with_llm
from langchain_agents import analyze_patients_with_langchain
from visualization import (
    plot_pulmonary_function_trends,
    create_lab_results_radar,
    create_patient_summary_table,
    create_risk_assessment_dashboard
)
from utils import get_session_id

# Sample patient data for testing
SAMPLE_PATIENT = {
    "id": "ILD-2023-045",
    "name": "Sample Patient",
    "case_date": "2023/04/11",
    "physician": "Dr. Smith",
    "diagnosis": "SLE overlapping RA with ILD",
    "imaging_diagnosis": "UIP pattern with honeycombing",
    "case_summary": "58-year-old female with progressive dyspnea, dry cough, and fatigue. History of SLE overlapping RA for 26 years with arthritis and positive autoimmune markers. ILD suspected after endoxan pulse therapy in 1995.",
    "medications": {
        "Bronchodilator": "Relvar ellipta QD (Fluticasone, Vilanterol ellipta)",
        "Immunosuppressive agent": "AZA 50mg TIW, Tofacitinib 11mg QD, Prednisolone 10mg QD",
        "Anti-fibrotic agent": "None",
        "Pulmonary hypertension agent": "None",
        "Others": "HCQ 200mg QD, montelukast 10mg HS, erythromycin 750mg QD"
    },
    "immunologic_profile": {
        "ANA": "1:1280",
        "SS-A": "> 240",
        "SS-B": "220",
        "RF": "85.9",
        "Myositis Ab": "Ku:++/ Ro-52:+++"
    },
    "biologic_markers": {
        "Ferritin": "9.81",
        "ESR": "27",
        "hs-CRP": "0.011",
        "NT-ProBNP": "78.67"
    },
    "pulmonary_tests": [
        {
            "date": "2023/01/15",
            "FVC": "51%",
            "FEV1": "50%",
            "FEV1/FVC": "82%",
            "DLCO": "40%"
        }
    ],
    "hrct": {
        "date": "2023/02/10",
        "findings": "Reticulation over periphery of right lower lobe and lower lobe with honeycombing pattern",
        "impression": "UIP should be considered, stable compared to previous CT"
    },
    "discussion_points": [
        {
            "question": "Is this patient's condition considered ILD?",
            "answer": "Yes, the patient has features consistent with ILD based on HRCT findings and pulmonary function tests."
        },
        {
            "question": "Does the patient have UIP pattern?",
            "answer": "Yes, the HRCT findings show reticulation with a peripheral distribution and honeycombing, which are consistent with a UIP pattern."
        },
        {
            "question": "Is there ongoing rheumatic disease activity?",
            "answer": "Yes, there are elevated autoimmune markers suggesting ongoing rheumatic disease activity."
        }
    ]
}

# Set page configuration
st.set_page_config(
    page_title="ILD Patient Analysis System",
    page_icon="🫁",
    layout="wide",
)

# Import the LLM manager
from llm_providers import llm_manager, OPENAI, OLLAMA

# Check for API keys and LLM availability
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
api_key_available = OPENAI_API_KEY is not None and len(OPENAI_API_KEY.strip()) > 0

# Initialize session state
if "patients_data" not in st.session_state:
    st.session_state.patients_data = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None
if "session_id" not in st.session_state:
    st.session_state.session_id = get_session_id()
if "api_key_warning_shown" not in st.session_state:
    st.session_state.api_key_warning_shown = False

# Main title
st.title("ILD Patient Analysis System")
st.write("A LangChain-powered multi-agent system for collaborative analysis of Interstitial Lung Disease patients")
st.markdown("### Featuring LLM-enhanced PDF processing and comprehensive multi-disciplinary analysis")

# Display LLM provider info
if not st.session_state.api_key_warning_shown:
    if api_key_available:
        st.success("✅ OpenAI API key is configured and available for use.")
    else:
        st.info("ℹ️ OpenAI API key is not configured. You can still use Ollama as a local LLM provider if you have it installed.")
    
    st.session_state.api_key_warning_shown = True

# Sidebar for uploading and managing files
with st.sidebar:
    st.header("Document Management")
    
    # LLM Provider Selection
    st.subheader("LLM Provider Settings")
    
    # Initialize session state variables for LLM provider
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = OPENAI if api_key_available else OLLAMA
        llm_manager.set_provider(st.session_state.llm_provider)
    
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = llm_manager.get_current_model()
    
    # Get available providers
    available_providers = llm_manager.get_available_providers()
    
    # Default to OpenAI if it's available
    if not available_providers and api_key_available:
        available_providers = [OPENAI]
    
    if available_providers:  
        # Provider selection
        provider_options = available_providers
        provider_display_names = {OPENAI: "OpenAI", OLLAMA: "Ollama (Local)"}
        
        selected_provider = st.selectbox(
            "Select LLM Provider",
            options=provider_options,
            format_func=lambda x: provider_display_names.get(x, x),
            key="provider_selectbox"
        )
        
        # Update provider if it changed
        if selected_provider != st.session_state.llm_provider:
            if llm_manager.set_provider(selected_provider):
                st.session_state.llm_provider = selected_provider
                st.session_state.llm_model = llm_manager.get_current_model()
                st.success(f"LLM provider set to {provider_display_names.get(selected_provider, selected_provider)}")
            else:
                st.error(f"Failed to set provider to {provider_display_names.get(selected_provider, selected_provider)}")
        
        # Model selection based on provider
        available_models = llm_manager.get_available_models(st.session_state.llm_provider)
        
        if available_models:
            selected_model = st.selectbox(
                "Select Model",
                options=available_models,
                index=available_models.index(st.session_state.llm_model) if st.session_state.llm_model in available_models else 0,
                key="model_selectbox"
            )
            
            # Update model if it changed
            if selected_model != st.session_state.llm_model:
                if llm_manager.set_model(st.session_state.llm_provider, selected_model):
                    st.session_state.llm_model = selected_model
                    st.success(f"Model set to {selected_model}")
                else:
                    st.error(f"Failed to set model to {selected_model}")
        
        # Additional settings for Ollama
        if st.session_state.llm_provider == OLLAMA:
            st.subheader("Ollama Settings")
            ollama_url = st.text_input("Ollama URL", value=llm_manager.ollama_url)
            
            if ollama_url != llm_manager.ollama_url:
                if st.button("Update Ollama URL"):
                    if llm_manager.set_ollama_url(ollama_url):
                        st.success(f"Ollama URL updated to {ollama_url}")
                    else:
                        st.error("Failed to update Ollama URL")
    else:
        st.warning("No LLM providers are available. Please configure either OpenAI API key or set up local Ollama.")
    
    # Document uploading
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader("Upload ILD patient document (PDF)", type="pdf")
    
    # Add a prominent button to use sample data for testing
    st.markdown("### Quick Test Option")
    st.markdown("Try the app without uploading a file:")
    if st.button("📋 Use Sample Patient Data", use_container_width=True):
        st.session_state.patients_data = [SAMPLE_PATIENT]
        st.info("Sample patient data loaded!")
        
        with st.spinner("Analyzing sample patient with multi-agent system..."):
            st.session_state.analysis_results = analyze_patients_with_langchain(st.session_state.patients_data)
            st.success("Multi-agent analysis complete!")
    
    if uploaded_file is not None:
        # Process the uploaded file
        with st.spinner("Processing document..."):
            # Save the uploaded file to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            
            try:
                # Process PDF using LLM-based extraction
                with st.status("Processing document...", expanded=True) as status:
                    st.write("Extracting text from PDF...")
                    pdf_text = extract_text_from_pdf(temp_file.name)
                    st.write("Text extracted successfully.")
                    
                    st.write("Using LLM to extract structured patient data...")
                    # Use the LLM-based processor to extract patient data
                    patients_data = process_pdf_with_llm(temp_file.name)
                    status.update(label="Document processed successfully!", state="complete")
                
                if patients_data:
                    # Debug info
                    st.text(f"Found {len(patients_data)} patient records in the PDF")
                    for i, p in enumerate(patients_data):
                        st.text(f"Patient {i+1}: ID={p.get('id', 'Unknown')}, Name={p.get('name', 'Unknown')}")
                        
                        # Ensure required fields exist for LangChain processing
                        if 'name' not in p:
                            patients_data[i]['name'] = f"Patient {i+1}"
                        if 'id' not in p:
                            patients_data[i]['id'] = f"ID-{i+1}"
                    
                    st.session_state.patients_data = patients_data
                    st.success(f"Successfully extracted {len(patients_data)} patient records")
                    
                    # Analyze patient data using LangChain multi-agent system
                    with st.spinner("Analyzing patient data with multi-agent system..."):
                        st.session_state.analysis_results = analyze_patients_with_langchain(patients_data)
                        st.success("Multi-agent analysis complete!")
                else:
                    st.error("No patient data could be extracted from the document")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)
    
    if st.session_state.patients_data:
        st.subheader("Patient Selection")
        # Handle cases where name or id might be missing
        patient_names = []
        for i, p in enumerate(st.session_state.patients_data):
            name = p.get('name', f"Patient {i+1}")
            id_value = p.get('id', f"ID-{i+1}")
            patient_names.append(f"{name} ({id_value})")
            
            # Ensure required fields exist for LangChain processing
            if 'name' not in p:
                st.session_state.patients_data[i]['name'] = f"Patient {i+1}"
            if 'id' not in p:
                st.session_state.patients_data[i]['id'] = f"ID-{i+1}"
                
        # Add option to view all patients in comparison view
        view_options = ["Single Patient View", "Multi-Patient Comparison"]
        view_mode = st.radio("Select view mode", view_options)
        
        if view_mode == "Single Patient View":
            selected_patient_idx = st.selectbox(
                "Select a patient to view details",
                range(len(patient_names)),
                format_func=lambda i: patient_names[i]
            )
            
            st.session_state.selected_patient = st.session_state.patients_data[selected_patient_idx]
            st.session_state.comparison_view = False
        else:
            st.session_state.comparison_view = True
            # No need to select a specific patient in comparison view

# Initialize the comparison view flag if not present
if "comparison_view" not in st.session_state:
    st.session_state.comparison_view = False

# Main content area
if st.session_state.comparison_view and st.session_state.patients_data:
    # Multi-Patient Comparison View
    st.header("Multi-Patient Comparison View")
    
    # Create tabs for different comparison sections
    comparison_tabs = st.tabs(["Basic Information", "Diagnosis & Findings", "Multi-Agent Analysis", "Risk Assessment"])
    
    # Tab 1: Basic Information Comparison
    with comparison_tabs[0]:
        st.subheader("Patient Information Comparison")
        
        # Create a DataFrame for basic patient info
        basic_info = []
        for patient in st.session_state.patients_data:
            info = {
                "ID": patient.get('id', 'N/A'),
                "Name": patient.get('name', 'N/A'),
                "Case Date": patient.get('case_date', 'N/A'),
                "Diagnosis": patient.get('diagnosis', 'N/A')
            }
            basic_info.append(info)
        
        # Display as a table
        st.table(pd.DataFrame(basic_info))
    
    # Tab 2: Diagnosis & Findings
    with comparison_tabs[1]:
        st.subheader("Diagnosis and HRCT Findings")
        
        findings = []
        for patient in st.session_state.patients_data:
            finding = {
                "ID": patient.get('id', 'N/A'),
                "Name": patient.get('name', 'N/A'),
                "Diagnosis": patient.get('diagnosis', 'N/A'),
                "Imaging Diagnosis": patient.get('imaging_diagnosis', 'N/A')
            }
            
            # Add HRCT date and impression if available
            if 'hrct' in patient and patient['hrct']:
                finding["HRCT Date"] = patient['hrct'].get('date', 'N/A')
                finding["HRCT Impression"] = patient['hrct'].get('impression', 'N/A')[:100] + "..." if len(patient['hrct'].get('impression', 'N/A')) > 100 else patient['hrct'].get('impression', 'N/A')
            
            findings.append(finding)
        
        # Display as a table
        st.table(pd.DataFrame(findings))
        
        # Show detailed HRCT findings in expandable sections
        st.subheader("Detailed HRCT Findings")
        for i, patient in enumerate(st.session_state.patients_data):
            with st.expander(f"{patient.get('name', f'Patient {i+1}')} - HRCT Details"):
                if 'hrct' in patient and patient['hrct']:
                    st.markdown(f"**Date:** {patient['hrct'].get('date', 'N/A')}")
                    st.markdown("**Findings:**")
                    st.write(patient['hrct'].get('findings', 'No detailed findings available'))
                    st.markdown("**Impression:**")
                    st.write(patient['hrct'].get('impression', 'No impression available'))
                else:
                    st.write("No HRCT findings available")
    
    # Tab 3: Multi-Agent Analysis Comparison
    with comparison_tabs[2]:
        st.subheader("Multi-Agent Analysis Comparison")
        
        # Show analysis results in expandable sections organized by patient
        for i, patient in enumerate(st.session_state.patients_data):
            analysis = next((a for a in st.session_state.analysis_results if a['patient_id'] == patient['id']), None)
            
            if analysis:
                with st.expander(f"{patient.get('name', f'Patient {i+1}')} - Analysis Summary"):
                    # Summary of key findings
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Diagnosis Analysis")
                        st.write(analysis.get('diagnosis_analysis', 'No diagnosis analysis available')[:200] + "..." if len(analysis.get('diagnosis_analysis', 'No diagnosis analysis available')) > 200 else analysis.get('diagnosis_analysis', 'No diagnosis analysis available'))
                    
                    with col2:
                        st.markdown("#### Progression Assessment")
                        st.write(analysis.get('progression_assessment', 'No progression assessment available')[:200] + "..." if len(analysis.get('progression_assessment', 'No progression assessment available')) > 200 else analysis.get('progression_assessment', 'No progression assessment available'))
                    
                    # Treatment recommendations
                    st.markdown("#### Treatment Recommendations")
                    st.write(analysis.get('treatment_recommendations', 'No treatment recommendations available')[:300] + "..." if len(analysis.get('treatment_recommendations', 'No treatment recommendations available')) > 300 else analysis.get('treatment_recommendations', 'No treatment recommendations available'))
                    
                    # View full analysis button
                    if st.button(f"View Full Analysis for {patient.get('name', f'Patient {i+1}')}"):
                        st.session_state.selected_patient = patient
                        st.session_state.comparison_view = False
                        st.rerun()
            else:
                st.warning(f"No analysis results available for {patient.get('name', f'Patient {i+1}')}")
    
    # Tab 4: Risk Assessment Comparison
    with comparison_tabs[3]:
        st.subheader("Risk Assessment Comparison")
        
        # Create a table comparing risk levels
        risk_comparison = []
        for patient in st.session_state.patients_data:
            analysis = next((a for a in st.session_state.analysis_results if a['patient_id'] == patient['id']), None)
            
            if analysis:
                # Get the specific questions from risk assessment if available
                questions = analysis.get('specific_questions', {})
                
                # Create a dictionary with all the key info
                risk_info = {
                    "ID": patient.get('id', 'N/A'),
                    "Name": patient.get('name', 'N/A'),
                    "Overall Risk": analysis.get('risk_level', 'Unknown'),
                    "Top Risk Factors": ", ".join(analysis.get('risk_factors', [])[:2]) if analysis.get('risk_factors') else "None identified"
                }
                risk_comparison.append(risk_info)
        
        # Display as a table
        st.table(pd.DataFrame(risk_comparison))
        
        # Create a comparison of the 8 specific questions
        st.subheader("Clinical Questions Comparison")
        
        # Define the questions for reference
        question_list = [
            "是否為 ILD",
            "是否為 Indeterminate",
            "是否為 UIP",
            "是否還有 NSIP pattern",
            "是否還有免風疾病活動性(activity) 病變",
            "是否 ILD 持續進展",
            "是否調整免疫治療藥物",
            "是否建議使用抗肺纖維化藥物"
        ]
        
        # Create DataFrame for questions comparison
        questions_data = []
        
        for patient in st.session_state.patients_data:
            analysis = next((a for a in st.session_state.analysis_results if a['patient_id'] == patient['id']), None)
            
            if analysis:
                # Get the specific questions from the multi-agent analysis
                specific_questions = analysis.get('specific_questions', {})
                
                # If no specific questions provided by the multi-agent system, create empty dict
                if not specific_questions:
                    specific_questions = {}
                
                # Create patient question data
                q_data = {"Patient": patient.get('name', 'Unknown')}
                
                # Add answers for each question
                for q in question_list:
                    q_data[q] = specific_questions.get(q, "否")
                    
                questions_data.append(q_data)
        
        if questions_data:
            # Create the DataFrame
            questions_df = pd.DataFrame(questions_data)
            
            # Display with colored cells based on yes/no
            # Format the DataFrame with colored text
            def highlight_yes_no(val):
                if val == "是":
                    return 'color: green; font-weight: bold'
                elif val == "否":
                    return 'color: red'
                else:
                    return ''
            
            # Apply the styling
            styled_df = questions_df.style.applymap(highlight_yes_no, subset=question_list)
            st.dataframe(styled_df)
        
        # Show individual risk dashboards
        st.subheader("Individual Risk Dashboards")
        
        for i, patient in enumerate(st.session_state.patients_data):
            analysis = next((a for a in st.session_state.analysis_results if a['patient_id'] == patient['id']), None)
            
            if analysis:
                with st.expander(f"{patient.get('name', f'Patient {i+1}')} - Risk Dashboard"):
                    try:
                        # Generate the risk assessment dashboard
                        risk_dashboard = create_risk_assessment_dashboard(patient, analysis)
                        
                        # Display the dashboard
                        st.pyplot(risk_dashboard)
                    except Exception as e:
                        st.error(f"Error creating risk dashboard: {str(e)}")
            else:
                st.warning(f"No risk assessment available for {patient.get('name', f'Patient {i+1}')}")
        
        # Add legend explanation
        with st.expander("Risk Level Color Code Legend"):
            st.markdown("""
            - 🟢 **Low Risk** (Green): Minimal concern, stable condition
            - 🟡 **Moderate Risk** (Yellow): Requires monitoring and possible intervention
            - 🔴 **High Risk** (Red): Significant concern, requires immediate attention
            - ⚪ **Unknown** (Gray): Insufficient data to determine risk level
            """)

elif st.session_state.selected_patient:
    patient = st.session_state.selected_patient
    analysis = next((a for a in st.session_state.analysis_results if a['patient_id'] == patient['id']), None)
    
    # Patient Overview
    st.header(f"Patient Overview: {patient['name']}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        st.markdown(f"**ID:** {patient['id']}")
        st.markdown(f"**Case Date:** {patient.get('case_date', 'N/A')}")
        st.markdown(f"**Physician:** {patient.get('physician', 'N/A')}")
        st.markdown(f"**Diagnosis:** {patient.get('diagnosis', 'N/A')}")
        st.markdown(f"**Radiological Findings:** {patient.get('imaging_diagnosis', 'N/A')}")
    
    with col2:
        st.subheader("Case Summary")
        st.write(patient.get('case_summary', 'No case summary available'))
    
    # Medication and Treatment
    st.subheader("Current Medication")
    if 'medications' in patient and patient['medications']:
        for category, meds in patient['medications'].items():
            st.markdown(f"**{category}:** {meds}")
    else:
        st.write("No medication information available")
    
    # Laboratory Results
    st.subheader("Laboratory Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Immunologic Profile")
        if 'immunologic_profile' in patient:
            st.table(pd.DataFrame(patient['immunologic_profile'].items(), 
                                  columns=['Test', 'Result']))
        else:
            st.write("No immunologic profile available")
    
    with col2:
        st.markdown("#### Biologic Markers")
        if 'biologic_markers' in patient:
            st.table(pd.DataFrame(patient['biologic_markers'].items(), 
                                  columns=['Marker', 'Value']))
        else:
            st.write("No biologic markers available")
    
    # Pulmonary Function Tests
    st.subheader("Pulmonary Function Tests")
    if 'pulmonary_tests' in patient and patient['pulmonary_tests']:
        # Create a table for PFT results
        pft_df = pd.DataFrame(patient['pulmonary_tests'])
        st.table(pft_df)
        
        # Create visualization for PFT metrics
        st.subheader("Pulmonary Function Visualization")
        if len(pft_df) >= 1:
            try:
                fig = plot_pulmonary_function_trends(pft_df)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not create visualization: {str(e)}")
                st.info("Note: Multiple PFT measurements over time provide better trend visualization.")
    else:
        st.write("No pulmonary function test results available")
    
    # HRCT Findings
    st.subheader("HRCT Findings")
    if 'hrct' in patient and patient['hrct']:
        st.markdown(f"**Date:** {patient['hrct'].get('date', 'N/A')}")
        st.markdown("**Findings:**")
        st.write(patient['hrct'].get('findings', 'No detailed findings available'))
        st.markdown("**Impression:**")
        st.write(patient['hrct'].get('impression', 'No impression available'))
    else:
        st.write("No HRCT findings available")
    
    # Analysis Results and Recommendations
    if analysis:
        st.header("Multi-Agent Analysis")
        
        # Create tabs for different analysis views
        analysis_tabs = st.tabs(["Specialist Impressions", "Discussion Details", "Recommendations"])
        
        # Tab 1: Specialist Impressions
        with analysis_tabs[0]:
            # Display document discussion points if available
            if 'discussion_points' in patient and patient['discussion_points']:
                st.subheader("Document Discussion Points")
                for i, point in enumerate(patient['discussion_points']):
                    st.markdown(f"**{i+1}. {point['question']}** {point['answer']}")
            
            # Display specialists' impressions from multi-agent system
            if 'specialist_impressions' in analysis:
                st.subheader("Specialist Impressions")
                specialists = analysis.get('specialist_impressions', {})
                
                # Create tabs for each specialist
                if specialists:
                    specialist_tabs = st.tabs(list(specialists.keys()))
                    for i, (specialist_type, impression) in enumerate(specialists.items()):
                        with specialist_tabs[i]:
                            st.write(impression)
        
        # Tab 2: Multi-Agent Discussion
        with analysis_tabs[1]:
            # Display multi-agent discussion
            if 'meeting_discussion' in analysis:
                st.subheader("Multi-Agent Discussion")
                discussions = analysis.get('meeting_discussion', {})
                
                # Create expandable sections for each discussion question
                if discussions:
                    for question, discussion in discussions.items():
                        with st.expander(question):
                            st.markdown("**Coordinator:**")
                            st.write(discussion.get('coordinator_prompt', 'No coordinator input'))
                            
                            for specialist, response in discussion.get('specialist_responses', {}).items():
                                st.markdown(f"**{specialist.title()}:**")
                                st.write(response)
            
            # Display meeting conclusion
            if 'meeting_conclusion' in analysis:
                st.subheader("Meeting Conclusion")
                st.write(analysis.get('meeting_conclusion', 'No conclusion available'))
        
        # Tab 3: Recommendations
        with analysis_tabs[2]:
            # Agent Recommendations
            st.subheader("Final Recommendations")
            
            # Display the 8 specific questions first
            st.markdown("#### Key Clinical Questions")
            specific_questions = analysis.get('specific_questions', {})
            if specific_questions:
                for question, answer in specific_questions.items():
                    answer_color = "green" if answer == "是" else "red"
                    st.markdown(f"**{question}:** <span style='color:{answer_color}'>{answer}</span>", unsafe_allow_html=True)
            else:
                # If no specific questions are available, display a message
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
                
                # Show a message that questions are not available
                st.info("Clinical question answers are being processed by the multi-agent system.")
                
                # Display default "No" answers
                for question in questions:
                    st.markdown(f"**{question}:** <span style='color:red'>否</span>", unsafe_allow_html=True)
            
            # Diagnosis Analysis
            st.markdown("#### Diagnosis Analysis")
            st.write(analysis.get('diagnosis_analysis', 'No diagnosis analysis available'))
            
            # Treatment Recommendations
            st.markdown("#### Treatment Recommendations")
            st.write(analysis.get('treatment_recommendations', 'No treatment recommendations available'))
            
            # Disease Progression Assessment
            st.markdown("#### Disease Progression Assessment")
            st.write(analysis.get('progression_assessment', 'No progression assessment available'))
                

    else:
        st.warning("Analysis results not available for this patient")
else:
    # Display welcome message when no patient is selected
    st.info("Please upload an ILD patient document and select a patient to view analysis")
    
    # Sample image showing app workflow
    st.subheader("How to use this application")
    st.markdown("""
    1. Upload a PDF document containing ILD patient information via the sidebar
    2. The system will extract text and use OpenAI to process patient data
    3. Each patient's information is structured with medical context
    4. The multi-agent system performs comprehensive analysis
    5. Review the results including the 8 key clinical questions
    
    The LangChain multi-agent system includes:
    - A coordinator agent that facilitates the discussion
    - Five specialist agents (pulmonologist, rheumatologist, radiologist, pathologist, cardiologist)
    - Collaborative discussion on eight key clinical questions
    - Comprehensive analysis with specialist perspectives
    
    The analysis results include:
    - Key clinical questions (是否為 ILD, 是否為 UIP, etc.) with clear Yes/No answers
    - Specialist impressions from each medical expert
    - Complete multi-disciplinary team discussion
    - Diagnosis verification and assessment
    - Treatment recommendations
    - Disease progression tracking
    """)
