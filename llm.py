import os 
import json
from openai import OpenAI
from retriever import top_retriever
import streamlit as st


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
api_key = st.secrets["OPENROUTER_API_KEY"]

LLM_model = 'openrouter/free'
client = OpenAI(
    base_url= OPENROUTER_BASE_URL,
    # api_key=os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY") // local 
    api_key= api_key
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "top_retriever",
            "description": "Searches for the keywords and returns the top 3 (default 3) matching documents with metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Search keywords or query string to look up in the vector database.",
                    },
                    # "top_k": {
                    #     "type": "integer",
                    #     "description": "Number of top results to return. Defaults to 3.",
                    #     "default": 3,
                    # },
                },
                "required": ["keywords"],
            },
        },
    }
]

def LLM_Chat(query:str):

    LLM_PROMPT = """Role & Safety Mandate
You are **RuleOut**, an AI Pediatric Clinical Intelligence Assistant for licensed healthcare professionals. You provide evidence-based clinical decision support—NOT autonomous diagnosis or treatment. Never replace clinical judgment. Prioritize patient safety, accuracy, and transparency.

### Core Rules & Hallucination Prevention
- **Knowledge Base First:** Always query the connected medical knowledge base. Prioritize pediatric guidelines, peer-reviewed literature, and textbooks.
- **Zero Fabrication:** Never invent patient data, dosages, citations, DOIs, or page numbers. If evidence or patient data is missing, state it explicitly.
- **Urgency First:** If findings suggest a life-threatening pediatric emergency, lead with the red-flag warning immediately before giving full analysis.
- **Uncertainty:** Distinguish confirmed findings from hypotheses, established guidelines from emerging data, and association from causation. Flag conflicting evidence.

### Pediatric & Medication Guardrails
- **Age-Specific Logic:** Account for pediatric physiology, developmental stage (neonate to adolescent), growth parameters, and age-adjusted vital/lab ranges. Never apply adult assumptions.
- **Dosing Safety:** Never guess doses. Only calculate doses when age, weight, indication, formulation/concentration, route, renal/hepatic status, and a verified reference are present. Show step-by-step math with units, check allergies/contraindications/max doses, and flag uncertainties for clinician verification.

### Clinical Workflow & Output Format
Analyze patient inputs (demographics, vitals, history, meds, allergies, exam, labs/imaging) and structure clinical responses using relevant sections below (omit inapplicable sections):
- **A. Case Summary:** Concise presentation overview.
- **B. Key Findings:** Pertinent positives, negatives, and abnormal results.
- **C. Clinical Interpretation:** Pathophysiological reasoning behind findings.
- **D. Differential Diagnoses:** Prioritized list with supporting/contradicting findings and distinguishing tests (never state as confirmed).
- **E. Clinical Considerations:** Evidence-based workup and management options for clinician review.
- **F. Red Flags:** Urgent warning signs and potential complications.
- **G. Evidence & References:** Exact titles, sections, or pages retrieved from the knowledge base.
- **H. Missing Information:** Critical gaps that could alter the assessment.

 **Scope Restriction:** Respond strictly to pediatric medicine and clinical healthcare queries. Politely decline and refuse to engage with any off-topic, non-medical, or general-purpose requests.

### Tone
Maintain an objective, precise, collegial tone. Match response depth to case complexity—concise for quick lookups, thorough for complex cases. Never give false reassurance.

                    """

    messages = [
        {
            "role": "system",
            "content": LLM_PROMPT,
        },
        {"role": "user", "content": query},
    ]
     
    response = client.chat.completions.create(
    model=LLM_model,
    
    messages= messages,
    tools=tools,
                                            )

    msg = response.choices[0].message
    # when function is requested
    if msg.tool_calls:
        messages.append(msg)

        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)

            # call function
            retriever_result = top_retriever(
                keywords=args['keywords'],
                # top_k=args.get("top_k",3)
            )

            # returning 
            messages.append({
                "role": "tool",
                "tool_call_id":tool_call.id,
                "content": json.dumps(retriever_result,default=str),
            })

        # LLM will synthesize the answer
        final_response = client.chat.completions.create(
            model=LLM_model,
            messages=messages,
        )
        return final_response.choices[0].message.content

    return msg.content

    
# query =  "What are the clinical features of bronchiolitis in infants?"
# query =  "hi what is 20+ 30 *67"

# questions = ['What are the clinical features of bronchiolitis in infants?',
#              "What are the common causes of bronchiolitis in infants?",
#             #  "What clinical findings suggest severe bronchiolitis?",
#             #  "How does RSV affect the lower respiratory tract in infants?",
#             # "What is the difference between bronchiolitis and bronchial asthma?",
#             # "What are the signs of respiratory distress in a young infant?",
#             # "What are the indications for hospital admission in an infant with bronchiolitis?",
#             # "What findings indicate that an infant with bronchiolitis is improving?",
#             # "What are the complications associated with severe bronchiolitis?",
#             # "What is the role of oxygen therapy in bronchiolitis?",
#             # "Why can young infants with bronchiolitis develop apnoea?",
#             # "What are the red flags in an infant presenting with cough and difficulty breathing?",
#             # "What are the common clinical features of croup in children?",
#             # "What are the feeding-related concerns in infants with respiratory distress?",
#             # "What is the difference between upper respiratory infection and bronchiolitis?",
#              ]

# for query in questions:

#     x = LLM_Chat(query=query)
#     print("=" * 60)
#     print(x)
#     print(">>" * 30)

# response = client.chat.completions.create(
#     model=LLM_model,
#     messages=[
#         {
#             "role" :"user", "content":"hello add 5+25"
#         }
#     ]
# )

# print(response.choices[0].message.content.strip())