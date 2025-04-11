# ILD Patient Analysis System

A multi-agent system for analyzing Interstitial Lung Disease (ILD) patients.

## Overview

This application provides a comprehensive platform for analyzing ILD patient data using a multi-agent approach. It allows medical professionals to upload patient documents, extracts structured information, and provides in-depth analysis and recommendations.

## Features

- Upload and process PDF documents containing ILD patient data
- Extract structured patient information including:
  - Patient demographics
  - Diagnoses
  - Laboratory results
  - Pulmonary function tests
  - HRCT findings
  - Treatment plans
- Multi-agent analysis for different aspects of patient care:
  - Diagnosis verification agent
  - Treatment recommendation agent
  - Disease progression assessment agent
  - Risk assessment agent
- Visualization of key metrics and trends
- Structured recommendations for patient management

## Technology Stack

- Streamlit: Web interface
- PDFMiner: PDF document parsing
- Pandas: Data manipulation
- Matplotlib: Data visualization
- OpenAI API: Multi-agent intelligent analysis
- Regular expressions: Text processing and data extraction

## Usage

1. Upload an ILD patient document (PDF format)
2. The system will extract patient information and analyze it
3. Select a patient to view detailed analysis
4. Review the multi-agent recommendations and visualizations

## Requirements

- Python 3.7+
- OpenAI API key (set as environment variable OPENAI_API_KEY)
- Internet connection for API access

## Running the Application

