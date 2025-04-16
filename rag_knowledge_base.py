"""
RAG Knowledge Base Module for ILD Analysis App

This module implements a Retrieval-Augmented Generation (RAG) system
for providing medical knowledge to the multi-agent ILD analysis system.
It creates and manages a vector database of medical knowledge about 
interstitial lung diseases (ILD) that agents can query to enhance their analysis.
"""

import os
import json
from typing import List, Dict, Any, Optional
import tempfile

# Try to import required libraries - if they fail, we'll still define RAG_AVAILABLE
try:
    import numpy as np
    import faiss
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    # If we get here, all required libraries are available
    RAG_AVAILABLE = True
except ImportError:
    # Missing dependencies
    RAG_AVAILABLE = False
    print("RAG dependencies not available - RAG functionality will be disabled")

# Medical knowledge documents for ILD
ILD_KNOWLEDGE = [
    # General ILD information
    {
        "title": "ILD Classification",
        "content": """
        Interstitial Lung Disease (ILD) encompasses a heterogeneous group of disorders characterized by inflammation and/or fibrosis of the lung parenchyma. 
        Major categories include:
        1. ILDs of known cause (e.g., drug-induced, CTD-ILD)
        2. Idiopathic interstitial pneumonias (including IPF)
        3. Granulomatous ILDs (e.g., sarcoidosis)
        4. Other rare ILDs
        
        Accurate classification is essential for prognosis and treatment decisions, requiring a multidisciplinary approach involving pulmonologists, radiologists, and pathologists.
        """
    },
    
    # UIP Pattern information
    {
        "title": "UIP Pattern",
        "content": """
        Usual Interstitial Pneumonia (UIP) is a histopathological and radiological pattern characterized by:
        
        Radiological features:
        - Peripheral and basal predominance
        - Reticular abnormalities
        - Honeycombing with or without traction bronchiectasis
        - Minimal ground-glass opacities
        - Patchy distribution
        
        Histopathological features:
        - Patchy involvement of the lung parenchyma
        - Fibrosis with architectural distortion
        - Temporal heterogeneity with fibroblastic foci
        - Honeycombing
        - Absence of features inconsistent with UIP
        
        UIP is the hallmark pattern of idiopathic pulmonary fibrosis (IPF) but can also be seen in CTD-ILD, chronic hypersensitivity pneumonitis, and drug-induced ILD.
        """
    },
    
    # NSIP Pattern information
    {
        "title": "NSIP Pattern",
        "content": """
        Nonspecific Interstitial Pneumonia (NSIP) is characterized by:
        
        Radiological features:
        - Bilateral, symmetric ground-glass opacities
        - Lower lung predominance
        - Subpleural sparing (in some cases)
        - Traction bronchiectasis
        - Minimal or absent honeycombing
        
        Histopathological features:
        - Temporally uniform involvement
        - Varying degrees of inflammation and fibrosis
        - Preserved alveolar architecture
        - Interstitial widening with lymphocyte/plasma cell infiltration
        
        NSIP is commonly associated with connective tissue diseases (especially polymyositis/dermatomyositis, rheumatoid arthritis, and systemic sclerosis), drug reactions, and hypersensitivity pneumonitis. It generally has a better prognosis than UIP.
        """
    },
    
    # CTD-ILD information
    {
        "title": "CTD-ILD",
        "content": """
        Connective Tissue Disease-associated ILD (CTD-ILD) refers to lung involvement in patients with autoimmune conditions. Key aspects include:
        
        Common causes:
        - Systemic sclerosis (most frequent and severe ILD)
        - Rheumatoid arthritis
        - Inflammatory myopathies (polymyositis/dermatomyositis)
        - Sjögren's syndrome
        - Systemic lupus erythematosus
        - Mixed connective tissue disease
        
        Radiological patterns:
        - NSIP pattern (most common)
        - UIP pattern
        - Organizing pneumonia
        - Lymphocytic interstitial pneumonia
        
        Management considerations:
        - Immunosuppressive therapy is the mainstay of treatment
        - Monitoring for disease progression with PFTs and imaging
        - Anti-fibrotic medications for progressive fibrosing phenotypes
        - Treatment of the underlying CTD
        """
    },
    
    # Treatment approaches
    {
        "title": "ILD Treatment",
        "content": """
        ILD treatment varies based on specific diagnosis, pattern, and progression. Key approaches include:
        
        1. Immunosuppressive therapy:
           - Corticosteroids (often first-line for inflammatory ILDs)
           - Cyclophosphamide (for severe CTD-ILD, especially with systemic sclerosis)
           - Mycophenolate mofetil (preferred for long-term management of CTD-ILD)
           - Azathioprine (alternative for maintenance therapy)
           - Rituximab (for refractory CTD-ILD)
           - Tacrolimus and cyclosporine (alternatives for refractory disease)
        
        2. Anti-fibrotic therapy:
           - Nintedanib (approved for IPF, progressive fibrosing ILDs including CTD-ILD)
           - Pirfenidone (approved for IPF, sometimes used off-label for other fibrosing ILDs)
           
        3. Combination therapy:
           - Emerging evidence supports combining immunosuppression and anti-fibrotics in some cases
           - Examples include mycophenolate plus nintedanib for progressive SSc-ILD
        
        4. Supportive care:
           - Oxygen therapy for hypoxemia
           - Pulmonary rehabilitation
           - Vaccination against respiratory pathogens
           - Management of comorbidities (GERD, pulmonary hypertension)
        
        5. Advanced options:
           - Lung transplantation for end-stage disease
           - Clinical trials for novel therapies
        """
    },
    
    # Disease progression criteria
    {
        "title": "ILD Progression",
        "content": """
        ILD progression assessment is crucial for management decisions. Key indicators include:
        
        1. Physiological markers:
           - Decline in FVC ≥5-10% over 6-12 months (depending on the underlying disease)
           - Decline in DLCO ≥15% over 6-12 months
           - Decrease in 6-minute walk distance ≥50 meters
           - Worsening of hypoxemia
        
        2. Radiological progression:
           - Increased extent of fibrosis on HRCT
           - New areas of honeycombing
           - Progression of traction bronchiectasis
           - Increased ground-glass opacities
        
        3. Clinical deterioration:
           - Worsening dyspnea (increase in mMRC score)
           - Increased cough
           - Decline in quality of life
           - Decreased exercise capacity
        
        4. Disease-specific considerations:
           - For CTD-ILD: activity of the underlying CTD
           - For IPF: any decline may represent progression due to the inevitably progressive nature
        
        The integration of these parameters through serial assessments (typically every 3-6 months) is essential for detecting progression early and adjusting treatment accordingly.
        """
    },
    
    # Clinical questions for multidisciplinary discussion
    {
        "title": "ILD Clinical Questions",
        "content": """
        Key clinical questions for ILD multidisciplinary discussion include:
        
        1. 是否為 ILD (Is it ILD?):
           Assessment based on clinical features, PFTs, and HRCT findings
           Differential diagnosis must exclude other causes of diffuse lung disease
        
        2. 是否為 Indeterminate (Is it indeterminate?):
           Cases where findings are inconclusive or mixed patterns are present
           May require additional testing or longitudinal follow-up
        
        3. 是否為 UIP (Is it UIP?):
           Evaluation of radiological and/or histological features consistent with UIP
           Consideration of technical quality and adequacy of specimens/images
        
        4. 是否還有 NSIP pattern (Is there NSIP pattern?):
           Assessment for features of NSIP which may coexist with other patterns
           Implications for prognosis and response to therapy
        
        5. 是否還有免風疾病活動性(activity) 病變 (Is there rheumatic disease activity?):
           Evaluation of autoimmune markers, clinical features, and extra-pulmonary manifestations
           Consideration of serology, joint symptoms, skin findings, etc.
        
        6. 是否 ILD 持續進展 (Is ILD progressing?):
           Review of serial PFTs, imaging, and symptoms for evidence of deterioration
           Application of disease-specific criteria for progression
        
        7. 是否調整免疫治療藥物 (Adjust immunotherapy medications?):
           Assessment of current treatment efficacy and tolerability
           Consideration of escalation, de-escalation, or switching therapy based on disease behavior
        
        8. 是否建議使用抗肺纖維化藥物 (Recommend anti-fibrotic medication?):
           Evaluation for indications for anti-fibrotic therapy
           Assessment of potential benefits vs. risks for the specific patient
        
        These questions form the framework for a structured multidisciplinary approach to diagnosis and management planning.
        """
    },
    
    # PFT interpretation
    {
        "title": "PFT Interpretation in ILD",
        "content": """
        Pulmonary Function Test (PFT) interpretation in ILD:
        
        1. Typical pattern:
           - Restrictive ventilatory defect: reduced TLC, FVC, FEV1 with normal or increased FEV1/FVC ratio
           - Reduced DLCO: often the most sensitive indicator of ILD
           - Reduced compliance
           - Hypoxemia, especially with exertion
        
        2. Significance of values:
           - Normal range: FVC >80% predicted, DLCO >80% predicted
           - Mild impairment: FVC 70-80%, DLCO 60-80%
           - Moderate impairment: FVC 50-70%, DLCO 40-60%
           - Severe impairment: FVC <50%, DLCO <40%
        
        3. Monitoring considerations:
           - FVC and DLCO should be monitored at 3-6 month intervals
           - Decline in FVC ≥5-10% is clinically significant
           - Decline in DLCO ≥15% is clinically significant
           - Stable parameters generally indicate stable disease
        
        4. Special considerations:
           - Combined pulmonary fibrosis and emphysema (CPFE):
             * May have preserved lung volumes despite significant fibrosis
             * Disproportionately reduced DLCO
           - Pulmonary hypertension:
             * Disproportionate reduction in DLCO
             * Significant oxygen desaturation with exercise
        
        5. Exercise testing:
           - 6-minute walk test valuable for monitoring
           - Oxygen desaturation during exercise may be present before resting abnormalities
           - Decrease in walk distance ≥50 meters is clinically significant
        """
    },
    
    # HRCT interpretation
    {
        "title": "HRCT Interpretation in ILD",
        "content": """
        High-Resolution Computed Tomography (HRCT) interpretation in ILD:
        
        1. UIP pattern (definite):
           - Basal and subpleural predominance
           - Reticular abnormalities
           - Honeycombing with or without traction bronchiectasis
           - Absence of features inconsistent with UIP
        
        2. Probable UIP pattern:
           - Basal and subpleural predominance
           - Reticular abnormalities
           - Traction bronchiectasis
           - Absence of honeycombing
           - Absence of features inconsistent with UIP
        
        3. Indeterminate for UIP:
           - Mixed or inconclusive features
           - Some features of UIP but with atypical distribution or other inconsistent findings
        
        4. NSIP pattern:
           - Ground-glass opacities
           - Reticular abnormalities
           - Traction bronchiectasis in more fibrotic cases
           - Basal predominance but often more symmetrical than UIP
           - Subpleural sparing in some cases
        
        5. Features inconsistent with UIP (suggesting alternative diagnosis):
           - Upper or mid-lung predominance
           - Peribronchovascular predominance
           - Extensive ground-glass opacities exceeding reticulation
           - Profuse micronodules
           - Discrete cysts away from areas of honeycombing
           - Air trapping or mosaic attenuation
           - Consolidation
        
        6. Quantitative assessment:
           - Disease extent: minimal (<10%), mild (10-25%), moderate (25-50%), severe (>50%)
           - Pattern proportions: proportion of ground-glass, reticulation, honeycombing
           - Progression: new areas of involvement, increased pattern extent, pattern evolution
        
        7. Distribution patterns:
           - Peripheral vs. central
           - Upper vs. lower lung
           - Patchy vs. diffuse
           - Unilateral vs. bilateral
        """
    }
]


class ILDKnowledgeBase:
    """
    A vector database for storing and retrieving ILD-related medical knowledge.
    This class provides RAG capabilities to the multi-agent system.
    """
    
    def __init__(self, use_openai=True):
        """
        Initialize the knowledge base.
        
        Args:
            use_openai (bool): Whether to use OpenAI embeddings or a local alternative
        """
        self.use_openai = use_openai
        self.vector_store = None
        self.documents = []
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Create and populate the vector database with ILD knowledge."""
        # Convert our knowledge data to Document objects
        for item in ILD_KNOWLEDGE:
            self.documents.append(
                Document(
                    page_content=item["content"],
                    metadata={"title": item["title"]}
                )
            )
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        split_docs = text_splitter.split_documents(self.documents)
        
        # Create embeddings and vector store
        try:
            # First try OpenAI embeddings
            if self.use_openai and os.environ.get("OPENAI_API_KEY"):
                embeddings = OpenAIEmbeddings()
                self.vector_store = FAISS.from_documents(split_docs, embeddings)
                print("Created knowledge base with OpenAI embeddings")
            else:
                # Fallback to simpler approach if no API key or not using OpenAI
                self._create_simple_vector_store(split_docs)
        except Exception as e:
            print(f"Error creating vector store with OpenAI: {e}")
            # Fallback to simpler approach
            self._create_simple_vector_store(split_docs)
    
    def _create_simple_vector_store(self, documents):
        """
        Create a simple vector store without requiring external embedding API.
        This is a fallback method that uses very basic embeddings.
        
        Args:
            documents (list): List of Document objects to include in the store
        """
        # For demonstration purposes, we'll use a very simple embedding method
        # In production, you would use a proper local embedding model like 
        # SentenceTransformers, but we're keeping it simple here
        
        # Simple token count-based embedding (not very effective but works without dependencies)
        def simple_embed(text):
            # Convert to lowercase and remove punctuation
            text = text.lower()
            for p in ',.:;!?()[]{}':
                text = text.replace(p, ' ')
            
            # Count occurrences of medical terms and create a simple vector
            terms = ["ild", "uip", "nsip", "fibrosis", "inflammation", "progression", 
                    "treatment", "immunosuppressive", "anti-fibrotic", "ctd", 
                    "rheumatic", "pulmonary", "hrct", "fvc", "dlco"]
            
            # Create a simple embedding based on term frequency
            embedding = []
            for term in terms:
                # Count occurrences and normalize by text length
                count = text.count(term)
                normalized = count / (len(text) / 100)  # Normalize per 100 chars
                embedding.append(normalized)
            
            return np.array(embedding, dtype=np.float32)
        
        # Create embeddings for all documents
        document_embeddings = []
        for doc in documents:
            embedding = simple_embed(doc.page_content)
            document_embeddings.append(embedding)
        
        # Create a FAISS index
        dimension = len(document_embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        embeddings_array = np.array(document_embeddings).astype(np.float32)
        index.add(embeddings_array)
        
        # Store the index, documents, and embedding function
        self.vector_store = {
            "index": index,
            "documents": documents,
            "embed_func": simple_embed
        }
        print("Created knowledge base with simple embeddings")
    
    def query(self, query_text, top_k=3):
        """
        Query the knowledge base for information relevant to the query.
        
        Args:
            query_text (str): The query text
            top_k (int): Number of top results to return
            
        Returns:
            str: Compiled knowledge relevant to the query
        """
        # If we're using OpenAI embeddings and FAISS
        if isinstance(self.vector_store, FAISS):
            results = self.vector_store.similarity_search(query_text, k=top_k)
            
            # Compile the results
            knowledge = ""
            for i, doc in enumerate(results):
                knowledge += f"\n--- From {doc.metadata.get('title', 'Medical Literature')} ---\n"
                knowledge += doc.page_content + "\n"
            
            return knowledge
            
        # If we're using our simple embedding approach
        elif isinstance(self.vector_store, dict):
            # Get the embedding for the query
            query_embedding = self.vector_store["embed_func"](query_text)
            query_embedding = np.array([query_embedding]).astype(np.float32)
            
            # Search the FAISS index
            distances, indices = self.vector_store["index"].search(query_embedding, top_k)
            
            # Compile the results
            knowledge = ""
            for i, idx in enumerate(indices[0]):
                if idx < len(self.vector_store["documents"]):
                    doc = self.vector_store["documents"][idx]
                    knowledge += f"\n--- From {doc.metadata.get('title', 'Medical Literature')} ---\n"
                    knowledge += doc.page_content + "\n"
            
            return knowledge
            
        # Fallback if vector store initialization failed
        else:
            # Just return the first few documents as a fallback
            knowledge = ""
            for i, doc in enumerate(self.documents[:top_k]):
                knowledge += f"\n--- From {doc.metadata.get('title', 'Medical Literature')} ---\n"
                knowledge += doc.page_content[:500] + "...\n"  # Truncate long documents
            
            return knowledge


# Function to add RAG to an existing tool
def enhance_tool_with_rag(tool, knowledge_base):
    """
    Enhance a LangChain tool with RAG capabilities by creating a wrapper
    that adds relevant knowledge to the query before passing it to the tool.
    
    Args:
        tool: A LangChain tool
        knowledge_base: ILDKnowledgeBase instance
        
    Returns:
        A new tool with RAG capabilities
    """
    from langchain.tools import BaseTool
    
    original_run = tool._run
    
    def enhanced_run(query):
        # Get relevant knowledge
        relevant_knowledge = knowledge_base.query(query)
        
        # Enhance query with knowledge
        enhanced_query = f"""
        Query: {query}
        
        Relevant medical knowledge to consider:
        {relevant_knowledge}
        
        Based on this information, please respond to the original query.
        """
        
        # Call the original tool with the enhanced query
        return original_run(enhanced_query)
    
    # Replace the run method with our enhanced version
    tool._run = enhanced_run
    
    return tool


if __name__ == "__main__":
    # Example usage
    kb = ILDKnowledgeBase(use_openai=True)
    result = kb.query("What are the characteristics of UIP pattern on HRCT?")
    print(result)
    
    # Example of adding RAG to a tool
    from langchain.tools import BaseTool
    
    class DummyTool(BaseTool):
        name = "dummy"
        description = "A dummy tool"
        
        def _run(self, query):
            return f"Processing query: {query}"
        
        async def _arun(self, query):
            return self._run(query)
    
    tool = DummyTool()
    rag_tool = enhance_tool_with_rag(tool, kb)
    result = rag_tool._run("How do I differentiate UIP from NSIP?")
    print(result)