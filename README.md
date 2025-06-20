# Recruitment Recommendation Agent 💼

![Recruitment Agent Demo](https://via.placeholder.com/800x400?text=Recruitment+Agent+Demo+GIF/Image)

> An AI-powered recruitment assistant that analyzes resumes, matches candidates to job descriptions, and provides actionable insights.

## Features ✨

- **Resume Processing**: Extract structured data from PDF/DOCX resumes
- **Candidate Matching**: AI-powered analysis of candidate-job fit with scoring
- **Smart Ranking**: Sort candidates by fit score with detailed profiles
- **Interactive Chat**: Ask questions about candidates and get AI-powered answers
- **Comprehensive Reporting**: Generate detailed recruitment reports
- **Export Results**: Download candidate rankings as CSV or full reports as text

## Technologies Used 🛠️

- **Streamlit**: For building the web application interface
- **Groq API**: For AI-powered resume parsing and candidate analysis
- **PDF Plumber**: For extracting text from PDF files
- **Python-docx**: For extracting text from DOCX files
- **Pandas**: For data manipulation and CSV export

## Installation and Setup ⚙️

### Prerequisites
- Python 3.8+
- Groq API key (free at [Groq Cloud](https://console.groq.com/))

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/recruitment-agent.git

# Navigate to project directory
cd recruitment-agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
Configuration
Create a .env file in the project root:

env
GROQ_API_KEY=your_groq_api_key_here
Usage 🚀
Start the application:

bash
streamlit run main.py
In the browser:

Upload job description and candidate resumes

View candidate rankings and detailed profiles

Chat with the recruitment assistant

Generate comprehensive reports

Export results as CSV or text files

File Structure 📂
text
recruitment-agent/
├── main.py                 # Main application code
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
├── .gitignore              # Git ignore file
└── README.md               # Project documentation
Contributing 🤝
Contributions are welcome! Please follow these steps:

Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

License 📄
This project is licensed under the MIT License - see the LICENSE file for details.

Note: This application requires a Groq API key which you can get for free at Groq Cloud. The free tier provides sufficient capacity for testing and small-scale usage.
