# 📋 HR Assist — AI-Powered Company HR Policy Bot

> A production-grade agentic AI chatbot that gives employees instant, accurate, and grounded answers to HR policy questions — 24/7, no HR inbox required.

Built as the **Day 13 Capstone** for the *Agentic AI Hands-On Course* by Dr. Kanthi Kiran Sirra.

---

## 🧠 What It Does

HR Assist lets employees ask natural-language questions about company HR policies — leave entitlements, payroll, reimbursements, appraisals, resignation, benefits, WFH rules, and more — and get accurate, policy-grounded answers with a faithfulness score attached to every response.

**Example queries it handles:**
- *"How many annual leave days do I get per year?"*
- *"What is the notice period for a confirmed employee?"*
- *"Can I carry forward unused leave to next year?"*
- *"If I resign today, when is my last working day?"*
- *"What expenses can I claim for reimbursement?"*
- *"What are the rules for working from home?"*

---

## ✅ Capstone Capabilities

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | LangGraph StateGraph (3+ nodes) | 8-node pipeline: memory → router → retrieve/skip/tool → answer → eval → save |
| 2 | ChromaDB RAG (10+ documents) | 12 HR policy docs embedded with `all-MiniLM-L6-v2`, top-3 retrieval |
| 3 | Conversation memory | `MemorySaver` + `thread_id`; rolling 6-message window |
| 4 | Self-reflection / eval loop | LLM-as-judge faithfulness scoring (0–1); retries up to 2× if score < 0.7 |
| 5 | Tool use | Leave calculator: accrued AL/CL, notice period end date, business days left |
| 6 | Deployment | Streamlit chat UI with metadata, stats sidebar, and escalation badges |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────┐    ┌─────────┐    ┌──────────┐
│  memory │───▶│  router │───▶│ retrieve │──┐
└─────────┘    └─────────┘    └──────────┘  │
                    │                         │
                    ├──────────▶ skip ────────┤
                    │                         │
                    └──────────▶ tool ────────┤
                                              ▼
                                        ┌─────────┐    ┌──────┐    ┌──────┐
                                        │ answer  │───▶│ eval │───▶│ save │───▶ END
                                        └─────────┘    └──────┘    └──────┘
                                              ▲              │
                                              └── retry ─────┘
                                           (if faith < 0.7)
```

### Node Descriptions

| Node | Role |
|------|------|
| `memory` | Updates rolling conversation window; extracts employee name if introduced |
| `router` | Classifies query → `retrieve` / `memory_only` / `tool` |
| `retrieve` | Vector similarity search over ChromaDB; returns top-3 chunks + topic labels |
| `skip` | Bypasses retrieval for conversational follow-ups |
| `tool` | Python `datetime` leave calculator — accrued leave, notice period, business days |
| `answer` | Generates grounded response from context only; detects HR escalation signals |
| `eval` | LLM-as-judge faithfulness score (0.0–1.0); triggers retry if below threshold |
| `save` | Appends final answer to conversation history |

---

## 📚 Knowledge Base

12 HR policy documents covering:

- 🏖️ Annual & Casual Leave (18 AL + 6 CL days/year, carry-forward, half-day rules)
- 🤒 Sick Leave (12 days/year, hospitalisation leave, medical certificate requirements)
- 🏠 Work From Home (hybrid model, eligibility, Monday/Friday restrictions)
- 💰 Payroll & Salary (disbursement dates, CTC structure, TDS, salary advances)
- 🧾 Reimbursements (travel, meals, km rates, approval workflow)
- 📊 Performance Appraisal (1–5 rating scale, increment bands, PIP process)
- 🆕 Onboarding & Probation (6-month probation, confirmation, leave during probation)
- 📝 Resignation & Notice Period (60-day notice, buy-out, garden leave, absconding)
- 🏥 Employee Benefits (mediclaim, PF, gratuity formula, life insurance, EAP)
- ⚖️ Code of Conduct & Disciplinary Process (4-step process, severe misconduct)
- 🛡️ POSH Policy (ICC, complaint process, retaliation protection)
- 📅 Public Holidays & Company Calendar (12 holidays, restricted holidays, year-end shutdown)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Groq · LLaMA 3.3 70B Versatile |
| Orchestration | LangGraph (`StateGraph`) |
| Memory | LangGraph `MemorySaver` |
| Vector Store | ChromaDB (in-memory) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM SDK | LangChain / `langchain-groq` |
| Tool | Python `datetime` module |
| UI | Streamlit |
| Config | `python-dotenv` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier works)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hr-assist.git
cd hr-assist
```

### 2. Install dependencies

```bash
pip install langchain-groq langchain-core langgraph chromadb \
            sentence-transformers streamlit python-dotenv
```

### 3. Set your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app

```bash
streamlit run capstone_streamlit.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
hr-assist/
├── agent.py                      # Core LangGraph agent (build_agent)
├── capstone_streamlit.py         # Streamlit UI
├── day13_capstone_hr_assist.ipynb # Development notebook
├── .env                          # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 🖥️ UI Overview

```
┌─────────────────────────────────────┬──────────────────────┐
│  📋 HR Assist — Company Policy Bot  │  📋 HR Assist        │
│                                     │  Your 24/7 Assistant │
│  ┌─────────────────────────────┐    │  ─────────────────── │
│  │ You: How many annual leave  │    │  ✅ 12 docs loaded   │
│  │ days do I get per year?     │    │  Session: a3f7c2b1   │
│  └─────────────────────────────┘    │  Queries: 3          │
│                                     │  Avg Faith: 🟢 0.94  │
│  ┌─────────────────────────────┐    │  ─────────────────── │
│  │ HR Assist: You are entitled │    │  💡 Try asking:      │
│  │ to 18 days of Annual Leave  │    │  [ How many AL days ]│
│  │ per calendar year...        │    │  [ Notice period... ]│
│  │                             │    │  [ Carry forward... ]│
│  │ Faith: 0.97 | Route: ret... │    │  ─────────────────── │
│  └─────────────────────────────┘    │  🗑️ New conversation │
│                                     │                      │
│  [ Ask about leave, payroll... ]    │                      │
└─────────────────────────────────────┴──────────────────────┘
```

**Sidebar features:**
- Document count loaded from ChromaDB
- Per-session query counter
- Colour-coded average faithfulness: 🟢 ≥0.80 · 🟡 ≥0.60 · 🔴 <0.60
- Six quick-prompt buttons for common HR queries
- One-click conversation reset

**Chat features:**
- Per-response metadata: faithfulness score, routing decision, source document
- 🔴 HR escalation badge when the bot detects queries needing human HR contact
- Full multi-turn memory within a session

---

## ⚙️ Configuration

Key constants in `agent.py`:

```python
FAITHFULNESS_THRESHOLD = 0.7   # Minimum score before answer retry
MAX_EVAL_RETRIES       = 2     # Max retry attempts in the eval loop
MODEL_NAME             = "llama-3.3-70b-versatile"
EMBED_MODEL            = "all-MiniLM-L6-v2"
```

---

## 🔮 Future Improvements

- **HRMS Integration** — Connect to Darwinbox/BambooHR to retrieve actual employee leave balances instead of theoretical accruals
- **Persistent Vector Store** — Replace in-memory ChromaDB with a persistent instance; add an HR-facing document upload UI
- **Multi-Language Support** — Multilingual embeddings for Hindi and regional language queries
- **HR Portal Ticketing** — Auto-raise pre-populated support tickets when escalation is detected
- **Voice Interface** — Whisper (STT) + ElevenLabs (TTS) for field/factory floor employees
- **Analytics Dashboard** — Log queries, routes, faithfulness scores, and escalation flags for HR insights

---

> **HR Assist** · Agentic AI  Capstone · Instructor : Dr. Kanthi Kiran Sirra