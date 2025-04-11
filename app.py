import streamlit as st
import os
import tempfile
import pandas as pd
import json
import matplotlib.pyplot as plt
import io

from pdf_processor import extract_text_from_pdf
from data_extractor import extract_patient_data
from langchain_agents import analyze_patients_with_langchain
from visualization import (
    plot_pulmonary_function_trends,
    create_lab_results_radar,
    create_patient_summary_table
)
from utils import get_session_id

# Set page configuration
st.set_page_config(
    page_title="ILD Patient Analysis System",
    page_icon="🫁",
    layout="wide",
)

# Initialize session state
if "patients_data" not in st.session_state:
    st.session_state.patients_data = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None
if "session_id" not in st.session_state:
    st.session_state.session_id = get_session_id()

# Main title
st.title("ILD Patient Analysis System")
st.write("A LangChain-powered multi-agent system for collaborative analysis of Interstitial Lung Disease patients")

# Sidebar for uploading and managing files
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload ILD patient document (PDF)", type="pdf")
    
    if uploaded_file is not None:
        # Process the uploaded file
        with st.spinner("Processing document..."):
            # Save the uploaded file to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            
            try:
                # Extract text from PDF
                pdf_text = extract_text_from_pdf(temp_file.name)
                
                # Extract structured patient data
                patients_data = extract_patient_data(pdf_text)
                
                if patients_data:
                    st.session_state.patients_data = patients_data
                    # Analyze patient data using LangChain multi-agent system
                    st.session_state.analysis_results = analyze_patients_with_langchain(patients_data)
                    st.success(f"Successfully processed {len(patients_data)} patient records")
                else:
                    st.error("No patient data could be extracted from the document")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)
    
    if st.session_state.patients_data:
        st.subheader("Patient Selection")
        patient_names = [f"{p['name']} ({p['id']})" for p in st.session_state.patients_data]
        selected_patient_idx = st.selectbox(
            "Select a patient to view details",
            range(len(patient_names)),
            format_func=lambda i: patient_names[i]
        )
        
        st.session_state.selected_patient = st.session_state.patients_data[selected_patient_idx]

# Main content area
if st.session_state.selected_patient:
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
        
        # Create visualization for PFT trends
        if len(pft_df) > 1:
            st.subheader("Pulmonary Function Trends")
            fig = plot_pulmonary_function_trends(pft_df)
            st.pyplot(fig)
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
        
        # Display document discussion points if available
        if 'discussion_points' in patient:
            st.subheader("Document Discussion Points")
            for i, point in enumerate(patient['discussion_points']):
                st.markdown(f"**{i+1}. {point['question']}** {point['answer']}")
        
        # Display specialists' impressions from multi-agent system
        if 'specialist_impressions' in analysis:
            st.subheader("Specialist Impressions")
            specialists = analysis.get('specialist_impressions', {})
            
            # Create tabs for each specialist
            if specialists:
                tabs = st.tabs(list(specialists.keys()))
                for i, (specialist_type, impression) in enumerate(specialists.items()):
                    with tabs[i]:
                        st.write(impression)
        
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
        
        # Agent Recommendations
        st.subheader("Final Recommendations")
        
        # Diagnosis Analysis
        st.markdown("#### Diagnosis Analysis")
        st.write(analysis.get('diagnosis_analysis', 'No diagnosis analysis available'))
        
        # Treatment Recommendations
        st.markdown("#### Treatment Recommendations")
        st.write(analysis.get('treatment_recommendations', 'No treatment recommendations available'))
        
        # Disease Progression Assessment
        st.markdown("#### Disease Progression Assessment")
        st.write(analysis.get('progression_assessment', 'No progression assessment available'))
        
        # Risk Assessment
        st.markdown("#### Risk Assessment")
        risk_level = analysis.get('risk_level', 'Unknown')
        risk_factors = analysis.get('risk_factors', [])
        
        st.markdown(f"**Risk Level:** {risk_level}")
        st.markdown("**Risk Factors:**")
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    else:
        st.warning("Analysis results not available for this patient")
else:
    # Display welcome message when no patient is selected
    st.info("Please upload an ILD patient document and select a patient to view analysis")
    
    # Sample image showing app workflow
    st.subheader("How to use this application")
    st.markdown("""
    1. Upload a PDF document containing ILD patient information via the sidebar
    2. The system will extract and analyze patient data automatically
    3. Select a patient from the dropdown to view detailed analysis
    4. Review the multi-agent analysis and recommendations
    
    The LangChain multi-agent system includes:
    - A coordinator agent that facilitates the discussion
    - Five specialist agents (pulmonologist, rheumatologist, radiologist, pathologist, cardiologist)
    - Collaborative discussion on eight key clinical questions
    - Comprehensive analysis with specialist perspectives
    
    The analysis results include:
    - Specialist impressions from each medical expert
    - Complete multi-disciplinary team discussion
    - Diagnosis verification and assessment
    - Treatment recommendations
    - Disease progression tracking
    - Risk level assessment
    - Visualization of key metrics
    """)
