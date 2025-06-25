import os
import json
import pandas as pd
import streamlit as st
from docx import Document
import pdfplumber
from groq import Groq
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import uuid

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Set page config
st.set_page_config(
    page_title="Recruitment Recommendation Agent",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .match-high {
        color: #2e7d32;
        font-weight: bold;
    }
    .match-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .match-low {
        color: #f44336;
        font-weight: bold;
    }
    .candidate-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSpinner > div {
        text-align: center;
        margin-top: 20px;
    }
    .similarity-high {
        background-color: #e8f5e9;
    }
    .similarity-medium {
        background-color: #fffde7;
    }
    .similarity-low {
        background-color: #ffebee;
    }
    .vector-status {
        padding: 5px 10px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-top: 5px;
    }
    .vector-online {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'resumes' not in st.session_state:
    st.session_state.resumes = []
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'processed_candidates' not in st.session_state:
    st.session_state.processed_candidates = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'rag_mode' not in st.session_state:
    st.session_state.rag_mode = False
if 'top_n' not in st.session_state:
    st.session_state.top_n = 5
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None
if 'vector_metadata' not in st.session_state:
    st.session_state.vector_metadata = []
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Convert NumPy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Core functions
def extract_resume_data(text: str) -> Dict:
    """Extract structured data from resume text using Groq API."""
    prompt = f"""
    Extract the following information from the resume below in JSON format with keys: 
    "name", "email", "phone", "skills" (list), "experience" (list of dicts with "title", "company", "duration", "description"), 
    "education" (list of dicts with "degree", "institution", "year"), "certifications" (list), "summary".
    
    Resume:
    {text[:15000]}  # Truncate to avoid token limit
    
    Return ONLY the JSON object, nothing else.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=3000
        )
        response = chat_completion.choices[0].message.content
        return json.loads(response)
    except Exception as e:
        st.error(f"Error parsing resume: {str(e)}")
        return {}

def analyze_candidate_fit(candidate_data: Dict, job_description: str) -> Dict:
    """Analyze how well a candidate fits the job description using Groq API."""
    prompt = f"""
    Analyze candidate fit for the role. Return JSON with:
    - score (0-100)
    - strengths (list)
    - red_flags (list)
    - justification
    
    Candidate: {json.dumps(candidate_data)}
    Job Description: {job_description}
    
    Return ONLY the JSON object, nothing else.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=3000
        )
        response = chat_completion.choices[0].message.content
        return json.loads(response)
    except Exception as e:
        st.error(f"Error analyzing candidate: {str(e)}")
        return {"score": 0, "strengths": [], "red_flags": [], "justification": ""}

def generate_report(candidates: List[Dict], job_desc: str) -> str:
    """Generate comprehensive report using Groq API."""
    # Convert all NumPy types to Python native types
    serializable_candidates = convert_numpy_types(candidates[:10])  # Limit to 10 candidates
    
    prompt = f"""
    Generate a comprehensive recruitment report comparing candidates for a job role.
    Include candidate rankings, strengths, weaknesses, and recommendations.
    
    Job Description:
    {job_desc}
    
    Candidates:
    {json.dumps(serializable_candidates)}
    
    Structure your report with:
    1. Executive Summary
    2. Candidate Comparison (table format)
    3. Detailed Analysis per Candidate
    4. Final Recommendations
    
    Return the report in Markdown format.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.4,
            max_tokens=4000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return ""

def answer_recruitment_question(query: str, candidates: List[Dict], job_desc: str) -> str:
    """Answer recruitment questions using Groq API."""
    # Convert all NumPy types to Python native types
    serializable_candidates = convert_numpy_types(candidates[:10])  # Limit to 10 candidates
    
    prompt = f"""
    You are an expert recruitment assistant. Answer the following question based on the candidates and job description.
    
    Question: {query}
    
    Job Description:
    {job_desc}
    
    Candidates:
    {json.dumps(serializable_candidates)}
    
    Provide a concise, accurate answer with relevant details.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",
            temperature=0.3,
            max_tokens=2000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error answering question: {str(e)}")
        return "I encountered an error. Please try again."

# Streamlit Helper Functions
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    return text

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file."""
    text = ""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
    return text

def process_resumes(uploaded_files, job_desc):
    """Process resumes without RAG"""
    parsed_resumes = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            text = extract_text_from_pdf(file) if file.type == "application/pdf" else extract_text_from_docx(file)
            if not text:
                st.warning(f"Skipped {file.name}: Couldn't extract text")
                continue
            
            # Extract resume data
            with st.spinner(f"Analyzing {file.name}..."):
                parsed_data = extract_resume_data(text)
                
                if parsed_data:
                    # Analyze candidate fit
                    analysis_data = analyze_candidate_fit(parsed_data, job_desc)
                    
                    candidate_data = {
                        **parsed_data,
                        **analysis_data,
                        "filename": file.name
                    }
                    parsed_resumes.append(candidate_data)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    return parsed_resumes

def initialize_vector_index():
    """Initialize FAISS index for vector storage"""
    try:
        # Load embedding model
        model = load_embedding_model()
        embedding_size = model.get_sentence_embedding_dimension()
        
        # Create FAISS index
        index = faiss.IndexFlatIP(embedding_size)
        st.session_state.vector_index = index
        st.session_state.vector_metadata = []
        st.session_state.embedding_model = model
        return True
    except Exception as e:
        st.error(f"Error initializing vector database: {str(e)}")
        return False

def process_resumes_with_rag(uploaded_files, job_desc, top_n=5):
    """Process resumes using RAG with FAISS vector database"""
    # Initialize FAISS if not already done
    if st.session_state.vector_index is None:
        if not initialize_vector_index():
            st.error("Failed to initialize vector database. Falling back to regular processing.")
            return process_resumes(uploaded_files, job_desc)
    
    model = st.session_state.embedding_model
    index = st.session_state.vector_index
    metadata = []
    embeddings = []
    
    # Process all resumes and store embeddings
    with st.spinner("Processing resumes for vector database..."):
        for file in uploaded_files:
            text = extract_text_from_pdf(file) if file.type == "application/pdf" else extract_text_from_docx(file)
            if not text:
                continue
                
            # Generate embedding
            embedding = model.encode(text)
            embeddings.append(embedding)
            metadata.append({
                "filename": file.name,
                "text": text[:10000]  # Store first 10k characters for processing
            })
    
    # Add embeddings to FAISS index
    if embeddings:
        embeddings_array = np.array(embeddings).astype('float32')
        index.add(embeddings_array)
        st.session_state.vector_metadata = metadata
    
    # Query top candidates based on job description
    parsed_resumes = []
    with st.spinner("Finding top candidates using semantic search..."):
        # Generate query embedding
        query_embedding = model.encode([job_desc])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search index
        distances, indices = index.search(query_embedding, min(top_n, len(embeddings)))
        
        # Process top candidates
        top_indices = indices[0]
        top_distances = distances[0]
        
        progress_bar = st.progress(0)
        for i, (idx, distance) in enumerate(zip(top_indices, top_distances)):
            if idx < 0 or idx >= len(metadata):  # Safety check for index bounds
                continue
                
            progress_bar.progress((i + 1) / len(top_indices))
            
            # Convert cosine distance to similarity score
            similarity = float(distance)  # Convert to native float
            
            # Get resume text from metadata
            resume_text = metadata[idx]["text"]
            filename = metadata[idx]["filename"]
            
            # Extract resume data
            parsed_data = extract_resume_data(resume_text)
            if parsed_data:
                # Analyze candidate fit
                analysis_data = analyze_candidate_fit(parsed_data, job_desc)
                
                candidate_data = {
                    **parsed_data,
                    **analysis_data,
                    "filename": filename,
                    "initial_similarity": similarity
                }
                parsed_resumes.append(candidate_data)
        
        progress_bar.empty()
    
    return parsed_resumes

# UI Components
def display_candidate_card(candidate: Dict, rank: int):
    """Display candidate information in a styled card."""
    # Determine similarity class
    similarity = candidate.get('initial_similarity', 0)
    if similarity > 0.7:
        sim_class = "similarity-high"
    elif similarity > 0.5:
        sim_class = "similarity-medium"
    else:
        sim_class = "similarity-low"
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            name = candidate.get('name', f'Candidate {rank+1}')
            st.subheader(f"#{rank + 1}: {name}")
            st.write(f"📧 {candidate.get('email', 'N/A')} | 📞 {candidate.get('phone', 'N/A')}")
            
            # Display similarity if in RAG mode
            if st.session_state.rag_mode:
                st.metric("Semantic Match", f"{similarity*100:.1f}%", 
                         help="Similarity score from vector database search")
            
            score = candidate.get('score', 0)
            if score >= 75:
                score_class = "match-high"
            elif score >= 50:
                score_class = "match-medium"
            else:
                score_class = "match-low"
            
            st.markdown(f"**Fit Score:** <span class='{score_class}'>{score}%</span>", unsafe_allow_html=True)
            justification = candidate.get('justification', '')
            if justification:
                st.caption(justification[:200] + "..." if len(justification) > 200 else justification)
            
        with col2:
            if st.button(f"View Details 👉", key=f"view_{rank}_{name}"):
                st.session_state.selected_candidate = candidate
                st.session_state.current_tab = "Candidate Details"
                st.rerun()
        
        with st.expander("Quick View"):
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**✅ Strengths**")
                for strength in candidate.get('strengths', [])[:3]:
                    st.markdown(f"- {strength}")
            with col4:
                st.markdown("**⚠️ Red Flags**")
                for flag in candidate.get('red_flags', [])[:3]:
                    st.markdown(f"- {flag}")

def display_candidate_details(candidate: Dict):
    """Display detailed view of a single candidate."""
    st.button("← Back to Ranking", on_click=lambda: setattr(st.session_state, 'current_tab', 'Ranking'))
    name = candidate.get('name', 'Unnamed Candidate')
    st.title(name)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Fit Score", f"{candidate.get('score', 0)}%", 
                 help=candidate.get('justification', ''))
        
    with col2:
        if st.session_state.rag_mode:
            similarity = candidate.get('initial_similarity', 0)
            st.metric("Semantic Match", f"{similarity*100:.1f}%", 
                     help="Similarity score from vector database search")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.container(border=True):
            st.subheader("Contact Information")
            st.write(f"📧 **Email:** {candidate.get('email', 'N/A')}")
            st.write(f"📞 **Phone:** {candidate.get('phone', 'N/A')}")
        
        with st.container(border=True):
            st.subheader("Skills")
            for skill in candidate.get('skills', [])[:10]:
                st.markdown(f"- {skill}")
        
        with st.container(border=True):
            st.subheader("Education")
            for edu in candidate.get('education', [])[:3]:
                st.write(f"🎓 **{edu.get('degree', '')}**")
                st.write(f"{edu.get('institution', '')} ({edu.get('year', '')})")
                st.write("")
    
    with col2:
        with st.container(border=True):
            st.subheader("Experience")
            for exp in candidate.get('experience', [])[:3]:
                st.write(f"💼 **{exp.get('title', '')}**")
                st.write(f"🏢 {exp.get('company', '')} | {exp.get('duration', '')}")
                st.write(f"{exp.get('description', '')}")
                st.write("")
        
        if candidate.get('certifications'):
            with st.container(border=True):
                st.subheader("Certifications")
                for cert in candidate.get('certifications', [])[:5]:
                    st.markdown(f"- {cert}")
    
    with st.container(border=True):
        st.subheader("Recruiter's Analysis")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Strengths for this Role**")
            for strength in candidate.get('strengths', []):
                st.markdown(f"- ✅ {strength}")
        with col4:
            st.markdown("**Potential Red Flags**")
            for flag in candidate.get('red_flags', []):
                st.markdown(f"- ⚠️ {flag}")

# Main App
def main():
    st.title("💼 Recruitment Recommendation Agent")
    
    # Initialize tabs
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Upload"
    
    # Sidebar
    with st.sidebar:
        st.header("Recruitment Assistant")
        st.write("Upload resumes and a job description to get started.")
        
        st.selectbox(
            "Navigation",
            ["Upload", "Ranking", "Candidate Details", "Chat", "Reports"],
            key="nav_select",
            index=["Upload", "Ranking", "Candidate Details", "Chat", "Reports"].index(st.session_state.current_tab)
        )
        
        if st.session_state.nav_select != st.session_state.current_tab:
            st.session_state.current_tab = st.session_state.nav_select
            st.rerun()
        
        st.divider()
        api_key = st.text_input("Groq API Key", type="password", help="Get your API key from Groq Cloud")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        
        # Vector DB status
        st.divider()
        st.subheader("Vector Database")
        if st.session_state.vector_index is not None:
            num_vectors = st.session_state.vector_index.ntotal
            st.markdown(f'<div class="vector-status vector-online">FAISS: Online ({num_vectors} resumes indexed)</div>', unsafe_allow_html=True)
            st.caption("Resume embeddings stored in FAISS vector database")
        else:
            st.markdown('<div class="vector-status">FAISS: Ready to initialize</div>', unsafe_allow_html=True)
            if st.button("Initialize Vector DB"):
                if initialize_vector_index():
                    st.success("Vector database initialized successfully!")
                    st.rerun()
        
        st.divider()
        st.subheader("RAG Configuration")
        st.session_state.rag_mode = st.checkbox("Enable RAG Processing", value=True,
                                              help="Use vector search to select top candidates")
        if st.session_state.rag_mode:
            st.session_state.top_n = st.slider("Number of top candidates to analyze", 1, 20, 5,
                                              help="How many candidates to select for detailed analysis")
        
        st.divider()
        st.caption("Built with ❤️ using Streamlit, Groq Cloud, and FAISS")

    # Upload Tab
    if st.session_state.current_tab == "Upload":
        st.header("Upload Resumes and Job Description")
        
        st.subheader("1. Job Description")
        job_desc = st.text_area("Paste the job description here:", height=200, key="job_desc_input")
        
        st.subheader("2. Candidate Resumes")
        uploaded_files = st.file_uploader(
            "Upload resumes (PDF or DOCX):", 
            type=["pdf", "docx"], 
            accept_multiple_files=True
        )
        
        col1, col2 = st.columns(2)
        if col1.button("Process Resumes", disabled=not (uploaded_files and job_desc)):
            with st.spinner("Analyzing resumes..."):
                st.session_state.rag_mode = False
                st.session_state.job_description = job_desc
                st.session_state.resumes = process_resumes(uploaded_files, job_desc)
                
                if st.session_state.resumes:
                    st.session_state.processed_candidates = sorted(
                        st.session_state.resumes,
                        key=lambda x: x.get('score', 0),
                        reverse=True
                    )
                    st.session_state.current_tab = "Ranking"
                    st.rerun()
        
        rag_disabled = not (uploaded_files and job_desc)
        if col2.button("Process with RAG", disabled=rag_disabled):
            with st.spinner("Using vector database to select top candidates..."):
                st.session_state.rag_mode = True
                st.session_state.job_description = job_desc
                st.session_state.resumes = process_resumes_with_rag(
                    uploaded_files, 
                    job_desc,
                    st.session_state.top_n
                )
                
                if st.session_state.resumes:
                    st.session_state.processed_candidates = sorted(
                        st.session_state.resumes,
                        key=lambda x: x.get('score', 0),
                        reverse=True
                    )
                    st.session_state.current_tab = "Ranking"
                    st.rerun()

    # Ranking Tab
    elif st.session_state.current_tab == "Ranking":
        st.header("Candidate Ranking")
        
        if not st.session_state.processed_candidates:
            st.warning("No candidates processed yet. Please upload resumes first.")
            st.session_state.current_tab = "Upload"
            st.rerun()
        
        # Display RAG info if used
        if st.session_state.rag_mode:
            st.success(f"RAG Processing: Selected top {st.session_state.top_n} candidates using vector database")
        
        avg_score = sum(c.get('score', 0) for c in st.session_state.processed_candidates) / len(st.session_state.processed_candidates)
        top_score = st.session_state.processed_candidates[0].get('score', 0)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates Analyzed", len(st.session_state.processed_candidates))
        col2.metric("Average Score", f"{avg_score:.1f}%")
        col3.metric("Top Score", f"{top_score}%")
        
        min_score = st.slider("Filter by minimum score:", 0, 100, 50)
        filtered_candidates = [c for c in st.session_state.processed_candidates if c.get('score', 0) >= min_score]
        
        st.subheader(f"Showing {len(filtered_candidates)} Candidates (Score ≥ {min_score}%)")
        
        for i, candidate in enumerate(filtered_candidates):
            display_candidate_card(candidate, i)
        
        st.divider()
        st.subheader("Export Results")
        
        export_data = []
        for candidate in st.session_state.processed_candidates:
            export_data.append({
                "Name": candidate.get('name', ''),
                "Email": candidate.get('email', ''),
                "Phone": candidate.get('phone', ''),
                "Score": candidate.get('score', 0),
                "Top Skills": ", ".join(candidate.get('skills', [])[:5]),
                "Experience Years": len(candidate.get('experience', [])),
                "Education": ", ".join([edu.get('degree', '') for edu in candidate.get('education', [])]),
                "Strengths": "; ".join(candidate.get('strengths', [])),
                "Red Flags": "; ".join(candidate.get('red_flags', []))
            })
        
        df = pd.DataFrame(export_data)
        
        col1, col2 = st.columns(2)
        col1.download_button(
            label="Download as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='candidate_ranking.csv',
            mime='text/csv'
        )
        
        if col2.button("Generate Full Report"):
            with st.spinner("Generating report..."):
                st.session_state.analysis_results = generate_report(
                    st.session_state.processed_candidates,
                    st.session_state.job_description
                )
                st.session_state.current_tab = "Reports"
                st.rerun()

    # Candidate Details Tab
    elif st.session_state.current_tab == "Candidate Details":
        if 'selected_candidate' in st.session_state:
            display_candidate_details(st.session_state.selected_candidate)
        else:
            st.warning("No candidate selected. Please go back to Ranking.")
            st.session_state.current_tab = "Ranking"
            st.rerun()

    # Chat Tab
    elif st.session_state.current_tab == "Chat":
        st.header("Chat with Recruitment Assistant")
        
        if not st.session_state.processed_candidates:
            st.warning("No candidates processed yet. Please upload resumes first.")
            st.session_state.current_tab = "Upload"
            st.rerun()
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about candidates..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
                response = answer_recruitment_question(
                    prompt,
                    st.session_state.processed_candidates,
                    st.session_state.job_description
                )
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Reports Tab
    elif st.session_state.current_tab == "Reports":
        st.header("Recruitment Summary Report")
        
        if not st.session_state.analysis_results:
            st.warning("No analysis results available. Please process resumes first.")
            st.session_state.current_tab = "Upload"
            st.rerun()
        
        st.markdown(st.session_state.analysis_results)
        
        st.download_button(
            label="Download Report as Text",
            data=st.session_state.analysis_results,
            file_name='recruitment_summary_report.txt',
            mime='text/plain'
        )

if __name__ == "__main__":
    main()