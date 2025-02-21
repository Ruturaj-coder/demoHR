import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load FAISS index
index = faiss.read_index("hr_policies.index")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load HR policies mapping
hr_policies = {}
with open("hr_policies.txt", "r") as f:
    for line in f.readlines():
        key, value = line.strip().split(": ", 1)
        hr_policies[key] = value

# Define counter-questions for vague queries
clarifications = {
    "leave": ["sick leave", "casual leave", "annual leave", "paternity leave", "bereavement leave"],
    "salary": ["salary structure", "tax deductions", "overtime pay"],
    "benefits": ["health insurance", "travel reimbursement", "retirement plan"],
    "code_of_conduct": ["dress code", "remote work policy", "workplace ethics"],
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or use ["http://localhost:3000"] for strict security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/ask")
def ask_hr_question(query: str):
    # Convert query to vector
    query_vector = np.array([embedder.encode(query)])

    # Search FAISS
    D, I = index.search(query_vector, k=1)
    best_match_key = list(hr_policies.keys())[I[0][0]]
    best_match_text = hr_policies[best_match_key]

    # Check if the query is vague (i.e., contains a general category but no specifics)
    for keyword, specific_options in clarifications.items():
        if keyword in query.lower():  # If the query is about a broad category
            if not any(option in query.lower() for option in specific_options):
                # If no specific leave type, ask for clarification
                return {"answer": f"Do you mean {', '.join(specific_options)}?"}

    return {"answer": best_match_text}
