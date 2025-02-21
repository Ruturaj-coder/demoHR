import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# HR Policies Database (Expanded)
hr_policies = {
    # Leave Policies
    "sick_leave": "Employees are entitled to 10 days of paid sick leave per year. A medical certificate is required for leaves exceeding 3 days.",
    "casual_leave": "Employees can take up to 12 days of casual leave annually, subject to manager approval.",
    "annual_leave": "Employees get 20 days of paid annual leave. Unused leaves can be carried forward up to 30 days.",
    "maternity_leave": "Female employees are eligible for 26 weeks of paid maternity leave, extendable by 8 weeks unpaid.",
    "paternity_leave": "Male employees get 10 days of paternity leave within 6 months of childbirth.",
    "bereavement_leave": "Employees can take up to 5 days of paid leave in case of the death of an immediate family member.",
    
    # Payroll & Salary
    "salary_structure": "The salary is divided into Basic Pay (40%), HRA (20%), Allowances (10%), and Performance Bonus (30%).",
    "overtime_pay": "Overtime is paid at 1.5x the hourly rate for extra hours worked beyond 9 hours per day or 48 hours per week.",
    "tax_deductions": "Income tax, professional tax, and provident fund contributions are deducted from the monthly salary.",
    
    # Benefits & Insurance
    "health_insurance": "Employees are covered under a group health insurance plan with up to ₹5,00,000 hospitalization coverage.",
    "retirement_plan": "A Provident Fund contribution of 12% of Basic Pay is deducted, with an equal contribution from the employer.",
    "travel_reimbursement": "Employees on official travel can claim flight, hotel, and daily expenses as per company policy.",
    
    # Code of Conduct
    "dress_code": "Employees should dress in business casual attire. Fridays are casual dress days.",
    "remote_work_policy": "Employees can work remotely up to 3 days per week, subject to manager approval.",
    "workplace_ethics": "Employees must maintain professionalism, avoid conflicts of interest, and follow workplace harassment policies.",
}

# Convert policies to vector embeddings
policy_texts = list(hr_policies.values())
policy_vectors = np.array(embedder.encode(policy_texts))

# Store vectors in FAISS
index = faiss.IndexFlatL2(policy_vectors.shape[1])
index.add(policy_vectors)

# Save FAISS index
faiss.write_index(index, "hr_policies.index")

# Save text data mapping
with open("hr_policies.txt", "w") as f:
    for key, policy in hr_policies.items():
        f.write(f"{key}: {policy}\n")

print("HR policies loaded into FAISS successfully!")
