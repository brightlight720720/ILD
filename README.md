# ILD Analysis App

An advanced multi-agent medical analysis platform specialized in Interstitial Lung Disease (ILD) patient document processing and comprehensive recommendation generation.

## Features

- **PDF Document Analysis**: Extract patient data from medical documents
- **Multi-Agent System**: Utilizes LangChain to coordinate multiple specialist AI agents
- **Comprehensive Analysis**: Combines insights from pulmonology, rheumatology, radiology, and more
- **Key Clinical Questions**: Provides clear answers to 8 specific clinical questions in Chinese:
  - 是否為 ILD (Is it ILD?)
  - 是否為 Indeterminate (Is it Indeterminate?)
  - 是否為 UIP (Is it UIP?)
  - 是否還有 NSIP pattern (Is there NSIP pattern?)
  - 是否還有免風疾病活動性(activity) 病變 (Is there rheumatic disease activity?)
  - 是否 ILD 持續進展 (Is ILD progressing?)
  - 是否調整免疫治療藥物 (Adjust immunotherapy medications?)
  - 是否建議使用抗肺纖維化藥物 (Recommend anti-fibrotic medication?)
- **Multi-Patient Comparison**: Compare analysis across multiple patients

## Technical Stack

- **Python**: Core programming language
- **Streamlit**: Web interface framework
- **LangChain**: Multi-agent orchestration
- **OpenAI API**: GPT-powered medical analysis
- **PDF Processing**: Extract and structure medical data
- **Matplotlib**: Data visualization

## Getting Started

1. Clone this repository
2. Install the required packages: `pip install -r dependencies.txt`
3. Set up your OpenAI API key as an environment variable: `export OPENAI_API_KEY=your_api_key`
4. Run the application: `streamlit run app.py`

## Usage

1. Upload a PDF containing ILD patient information
2. The system extracts and structures the data
3. The multi-agent system analyzes the patient information
4. Review the analysis including the 8 key clinical questions
5. Compare multiple patients in the comparison view

## System Requirements

- Python 3.9+
- OpenAI API key
- 4GB+ RAM recommended for processing large documents