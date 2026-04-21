"""
agent.py — HR Assist Shared Agent Module
-----------------------------------------
Import this module in capstone_streamlit.py or any other deployment.
Usage:
    from agent import build_agent
    app, embedder, collection = build_agent()
    result = app.invoke({"question": "..."}, config={"configurable": {"thread_id": "abc"}})
"""

import os
from typing import TypedDict, List
from datetime import datetime, date

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2
MODEL_NAME             = "llama-3.3-70b-versatile"
EMBED_MODEL            = "all-MiniLM-L6-v2"

# ── Knowledge Base Documents ───────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Annual and Casual Leave Policy",
        "text": """Every confirmed employee is entitled to 18 days of Annual Leave (AL) per calendar year,
accrued at 1.5 days per month. Annual leave can be carried forward up to a maximum of 9 days into the
next calendar year; any balance above 9 days lapses on December 31st.
Casual Leave (CL) is 6 days per calendar year and cannot be carried forward or encashed. CL is meant
for unplanned short absences and must not be taken for more than 3 consecutive days.
Application rules: Annual leave must be applied at least 3 working days in advance through the HR portal.
Casual leave can be applied on the same day if it is an emergency. Manager approval is mandatory for AL;
CL approval is at the reporting manager's discretion.
Leave during notice period: Annual leave cannot be taken during the notice period without explicit HR
approval. Any unused AL at the time of resignation will be encashed at the basic daily rate.
Public holidays falling within a leave period are not counted as leave days.
Half-day leave: Both AL and CL can be taken as half days (morning or afternoon), which deducts 0.5 days
from the respective balance."""
    },
    {
        "id": "doc_002",
        "topic": "Sick Leave and Medical Leave Policy",
        "text": """Employees are entitled to 12 days of Sick Leave (SL) per calendar year. Sick leave does not
carry forward and cannot be encashed. It is non-transferable.
Medical certificate requirement: For absences of 3 or more consecutive days, a medical certificate from
a registered medical practitioner must be submitted to HR within 2 working days of returning to work.
Failure to submit documentation may result in the absence being treated as Leave Without Pay (LWP).
Extended illness: If an employee exhausts sick leave due to a serious illness, they may apply for
Special Medical Leave of up to 30 additional days per year with supporting hospital documentation.
This requires HR Head approval and is not guaranteed.
Hospitalisation leave: Employees who are hospitalised are entitled to up to 60 days of hospitalisation
leave per year, separate from the standard 12-day sick leave entitlement. This requires discharge
summary and hospital bills as documentation.
Abuse of sick leave: Patterns of sick leave taken immediately before or after weekends and public
holidays will be flagged and may be investigated by HR. Repeated abuse may result in disciplinary action."""
    },
    {
        "id": "doc_003",
        "topic": "Work From Home and Remote Work Policy",
        "text": """The company operates a hybrid work model. Employees in eligible roles may work from home
up to 2 days per week, subject to manager approval and business requirements.
Eligibility: Work from home is available to confirmed employees who have completed their probation period
(6 months). Employees on a Performance Improvement Plan (PIP) are not eligible for WFH until the PIP
is successfully closed.
WFH day rules: WFH days cannot be taken on Mondays, Fridays, or the day before or after a public holiday
without explicit manager approval, to prevent extended long weekends.
Equipment and connectivity: The company provides a laptop. Employees are responsible for their own
internet connection when working from home. The IT helpdesk provides remote support.
Work expectations during WFH: Employees must be reachable on Teams/Slack during core hours (10am–5pm),
attend all scheduled calls, and maintain the same output standards as in-office days.
Full remote work: Full-time remote arrangements are not standard policy. Exceptions require VP-level
approval and are reviewed every 6 months. Location change for remote work requires separate approval."""
    },
    {
        "id": "doc_004",
        "topic": "Payroll and Salary Disbursement Policy",
        "text": """Salaries are processed on the last working day of each month. In months where the last
working day falls on a weekend or public holiday, salaries are disbursed on the preceding working day.
Salary structure: The total Cost to Company (CTC) is split as follows — Basic (40% of CTC), House Rent
Allowance (20% of Basic), Special Allowance (variable), Provident Fund contribution (12% of Basic by
employer), and annual performance bonus (target 10–20% of CTC, paid in April).
Payslips: Digital payslips are available on the HR portal by the 2nd of each following month. Employees
must report any payroll discrepancy within 30 days of the payslip date.
Tax deduction: TDS (Tax Deducted at Source) is calculated based on the tax declaration submitted at the
start of the financial year (April). Employees must submit investment proof by January 31st to avoid
excess TDS deductions in the last quarter.
Salary advances: Employees may apply for a salary advance of up to 50% of one month's gross salary once
per financial year. Repayment is deducted over the following 3 months. HR Head approval is required."""
    },
    {
        "id": "doc_005",
        "topic": "Reimbursement and Expense Claims Policy",
        "text": """Employees may claim reimbursement for approved business expenses incurred on behalf of
the company, including travel, accommodation, client entertainment, and training materials.
Submission deadline: Expense claims must be submitted within 30 days of incurring the expense. Claims
submitted after 30 days will not be reimbursed without Finance Head approval.
Travel policy: For domestic travel, economy class flights and standard hotel accommodation are the default.
Business class requires VP approval. Daily meal allowance during travel is capped at INR 1,500 per day.
Local travel: Cab/auto fares for official meetings outside the office are reimbursable with receipts.
Personal vehicle use is reimbursed at INR 8 per km for two-wheelers and INR 12 per km for four-wheelers.
Approval workflow: Claims under INR 5,000 require manager approval only. Claims above INR 5,000 require
both manager and Finance approval. Payment is processed in the next payroll cycle after approval.
Non-reimbursable items: Alcohol, personal entertainment, fines, personal grooming, and gifts to employees
are not reimbursable under any circumstance."""
    },
    {
        "id": "doc_006",
        "topic": "Performance Review and Appraisal Cycle",
        "text": """The company follows an annual performance review cycle running from April 1st to March 31st,
aligned with the financial year.
Timeline: Goal-setting happens in April. Mid-year check-in in October. Final appraisal discussions in
March. Increment and promotion letters are issued in April.
Rating scale: Performance is rated on a 5-point scale — 1 (Below Expectations), 2 (Needs Improvement),
3 (Meets Expectations), 4 (Exceeds Expectations), 5 (Outstanding). A rating of 3 or above is required
to be eligible for a salary increment.
Increment bands: Rating 3 → 8–10% increment. Rating 4 → 11–15%. Rating 5 → 16–20%. Rating 1 or 2 →
no increment and mandatory Performance Improvement Plan (PIP).
PIP process: A PIP runs for 60–90 days with clear measurable goals. If the employee meets PIP targets,
they return to normal standing. If not, separation proceedings may begin.
Promotion criteria: Promotions require a Rating 4 or 5 for two consecutive years and manager nomination.
All promotions are reviewed by the Promotions Committee in February."""
    },
    {
        "id": "doc_007",
        "topic": "Onboarding and Probation Policy",
        "text": """All new employees undergo a 6-month probation period starting from their date of joining.
During probation, the notice period is 2 weeks on either side (employee or employer).
Onboarding program: The first week is structured onboarding — IT setup, policy induction, team introductions,
and a mandatory HR orientation session. Completion of the orientation is tracked in the HR portal.
Probation review: At the 5-month mark, the reporting manager submits a probation assessment form. HR
schedules a confirmation meeting. Employees who meet performance expectations are confirmed in writing.
Non-confirmation: If performance is unsatisfactory during probation, the probation may be extended by
up to 3 months or the employment may be terminated with 2 weeks' notice. The employee will be informed
in writing with specific feedback.
Leave during probation: Probationary employees are not eligible for Annual Leave. They accrue CL and SL
from day one but may not carry forward any balance if they leave before confirmation.
Benefits during probation: Health insurance and PF contributions begin from day one. Variable pay and
performance bonus eligibility begins after confirmation."""
    },
    {
        "id": "doc_008",
        "topic": "Resignation and Notice Period Policy",
        "text": """The standard notice period for confirmed employees is 60 days (2 calendar months), unless
a different period is specified in the individual employment contract.
How to resign: The employee must submit a formal written resignation to their reporting manager and HR
via the HR portal or email. Verbal resignations are not accepted.
Notice period buy-out: Employees may request early release by paying a notice period buy-out equivalent
to their basic salary for the remaining notice days. This requires HR Head approval and is not guaranteed.
Garden leave: The company reserves the right to place an employee on garden leave during the notice period,
meaning they are paid but not required (or allowed) to work. This is typically applied for senior or
sensitive roles.
Exit process: During the notice period, the employee must complete a knowledge transfer, return all
company property (laptop, access cards, documents), and obtain clearance from IT, Finance, and Admin.
The final settlement (pending salary, earned leave encashment, and PF paperwork) is processed within
30 days of the last working day.
Absconding: Employees who stop reporting to work without resignation or approval will be marked as
absconded. HR will send three written notices. If no response is received, the employment is terminated
for misconduct and the full notice pay will be recovered."""
    },
    {
        "id": "doc_009",
        "topic": "Employee Benefits — Insurance, PF, and Gratuity",
        "text": """The company provides the following benefits to all confirmed employees (and probationers
where noted):
Health Insurance: Group mediclaim policy covering employee, spouse, and up to 2 dependent children.
Sum insured: INR 3 lakhs per family per year. Coverage starts from day one. Pre-existing conditions are
covered after a 12-month waiting period. Employees may top up coverage at their own cost.
Provident Fund (PF): Both employee and employer contribute 12% of basic salary to the EPF account
each month, starting from day one. PF is withdrawable after 5 years of continuous service for full
tax-free benefit. Employees can check their PF balance on the EPFO member portal.
Gratuity: Employees are eligible for gratuity after completing 5 years of continuous service.
Formula: (Last drawn basic salary × 15 × number of years of service) / 26.
Life Insurance: Group term life cover of INR 20 lakhs for all employees, at no cost to the employee.
This is active from the date of confirmation.
Employee Assistance Program (EAP): Free and confidential counselling sessions (up to 6 per year) are
available through the company's EAP provider. Contact HR for the referral process."""
    },
    {
        "id": "doc_010",
        "topic": "Code of Conduct and Disciplinary Process",
        "text": """All employees are expected to maintain professional conduct at all times, both in the
workplace and in any external setting where they represent the company.
Key conduct obligations: Treat all colleagues, clients, and vendors with respect. Maintain confidentiality
of company information. Avoid conflicts of interest — declare any personal interest in a business decision
to your manager and HR. Do not accept gifts above INR 1,000 in value from vendors or clients.
Prohibited conduct: Harassment, discrimination, bullying, fraud, theft, data breach, substance abuse on
company premises, and misrepresentation of credentials or expenses are all grounds for immediate
disciplinary action.
Disciplinary process: Step 1 — Verbal warning (documented by manager). Step 2 — Written warning issued
by HR. Step 3 — Final written warning with improvement conditions. Step 4 — Termination for cause.
Severe misconduct (fraud, harassment, data theft) may result in immediate termination without prior
warnings, following an internal investigation.
Grievance redressal: Employees who feel they have been treated unfairly may raise a formal grievance
with HR within 30 days of the incident. HR will investigate and respond in writing within 15 working days."""
    },
    {
        "id": "doc_011",
        "topic": "Anti-Harassment and POSH Policy",
        "text": """The company is committed to providing a safe, respectful, and inclusive workplace for all
employees, in compliance with the Prevention of Sexual Harassment (POSH) Act, 2013.
Definition: Sexual harassment includes unwelcome physical contact, sexually coloured remarks, demands
for sexual favours, showing pornographic material, or any other unwelcome conduct of a sexual nature —
whether in person, online, or at work-related events outside the office.
Internal Complaints Committee (ICC): The company has a constituted ICC as required by law. The ICC
Chairperson is a senior woman employee. Contact details for the ICC are posted on the HR portal.
Complaint process: A written complaint must be submitted to the ICC within 3 months of the incident.
The ICC will complete an inquiry within 90 days and submit a report to management. Both parties have
the right to be heard. Confidentiality is maintained throughout.
Protection against retaliation: No employee who files a complaint in good faith will face adverse
employment action. Retaliation is itself a disciplinary offence.
Annual training: All employees must complete mandatory POSH awareness training once per year. Completion
is tracked in the HR portal. Non-completion may affect performance ratings."""
    },
    {
        "id": "doc_012",
        "topic": "Public Holidays and Company Calendar",
        "text": """The company observes 12 public holidays per calendar year. The official holiday list is
published on the HR portal in December for the following year.
National holidays (mandatory, no opt-out): Republic Day (January 26), Independence Day (August 15),
Gandhi Jayanti (October 2).
Restricted holidays: In addition to the 12 fixed holidays, employees may choose 2 Restricted Holidays
(RH) per year from a list of 10 regional and religious holidays published by HR. RH must be applied
for in advance through the HR portal.
Holiday falling on weekend: If a public holiday falls on a Saturday or Sunday, a compensatory day off
is granted on the nearest working day, as notified by HR.
Office shutdown: The company observes a mandatory year-end shutdown from December 26 to December 31.
These 4 days (excluding December 25 Christmas holiday) are deducted from the employee's Annual Leave
balance. Employees with insufficient AL balance will have these days treated as LWP.
Working on holidays: Employees required to work on a public holiday will receive a compensatory day off
to be taken within 60 days, subject to manager approval."""
    },
]


# ── State ──────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:            str
    messages:            List[dict]
    route:               str
    retrieved:           str
    sources:             List[str]
    tool_result:         str
    answer:              str
    faithfulness:        float
    eval_retries:        int
    employee_name:       str
    department:          str
    leave_balance:       float
    policy_references:   List[str]
    escalate_to_hr:      bool


# ── Node Functions ─────────────────────────────────────────
def make_nodes(llm, embedder, collection):

    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        updates = {"messages": msgs}
        # Extract employee name if introduced
        q = state["question"].lower()
        if "my name is" in q:
            parts = q.split("my name is")
            name = parts[1].strip().split()[0].capitalize()
            updates["employee_name"] = name
        return updates

    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent   = "; ".join(m["role"] + ": " + m["content"][:60] for m in messages[-3:-1]) or "none"

        prompt = f"""You are a router for HR Assist, a company HR policy assistant.

Available options:
- retrieve: search the HR policy knowledge base for questions about leave, payroll, reimbursements, appraisals, probation, resignation, benefits, conduct, POSH, or holidays
- memory_only: answer from conversation history (e.g. 'what did you just say?', 'remind me of that policy', follow-up clarification questions)
- tool: use the leave calculator tool (use when user asks to calculate remaining leave days, leave balance, days left in notice period, or any date/number arithmetic involving HR policies)

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

        response = llm.invoke(prompt)
        decision = response.content.strip().lower()
        if "memory" in decision:   decision = "memory_only"
        elif "tool" in decision:   decision = "tool"
        else:                      decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state: CapstoneState) -> dict:
        q_emb   = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks  = results["documents"][0]
        topics  = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(state: CapstoneState) -> dict:
        question = state["question"].lower()
        today    = date.today()
        try:
            result_lines = [f"LEAVE CALCULATOR — Today's date: {today.strftime('%B %d, %Y')}"]

            # Remaining leave days in the year
            day_of_year   = today.timetuple().tm_yday
            days_in_year  = 366 if (today.year % 4 == 0) else 365
            month_elapsed = today.month - 1 + today.day / 30
            al_accrued    = round(min(month_elapsed * 1.5, 18), 1)
            result_lines.append(f"Annual Leave accrued so far this year: {al_accrued} days (out of 18)")

            cl_accrued = round(min(month_elapsed * 0.5, 6), 1)
            result_lines.append(f"Casual Leave accrued so far this year: {cl_accrued} days (out of 6)")

            # Notice period calculation
            if "notice" in question or "last day" in question or "last working" in question:
                result_lines.append("Notice period: Standard notice is 60 calendar days from resignation date.")
                from datetime import timedelta
                notice_end = today + timedelta(days=60)
                result_lines.append(f"If resigning today ({today.strftime('%B %d, %Y')}), last working day would be: {notice_end.strftime('%B %d, %Y')}")

            # Days left in current year
            from datetime import date as d
            year_end      = d(today.year, 12, 31)
            days_remaining = (year_end - today).days
            result_lines.append(f"Working days remaining in {today.year}: approx {round(days_remaining * 5/7)} business days ({days_remaining} calendar days)")

            tool_result = "\n".join(result_lines)
        except Exception as e:
            tool_result = f"Leave calculation error: {str(e)}. Please contact HR at hr@company.com for manual calculation."
        return {"tool_result": tool_result}

    def answer_node(state: CapstoneState) -> dict:
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)

        context_parts = []
        if retrieved:   context_parts.append(f"RECRUITMENT KNOWLEDGE BASE:\n{retrieved}")
        if tool_result: context_parts.append(f"LEAVE CALCULATION DATA:\n{tool_result}")
        context = "\n\n".join(context_parts)

        if context:
            system_content = f"""You are HR Assist, a friendly and accurate company HR policy assistant.
You help employees understand their leave entitlements, payroll, reimbursements, appraisals, benefits,
resignation process, code of conduct, and all other HR policies.

Answer using ONLY the information provided in the context below.
Be clear, empathetic, and specific — employees need straightforward answers they can act on.
If the answer is not in the context, say: I don't have that information in the policy handbook.
Please contact HR at hr@company.com or raise a ticket on the HR portal for further help.
Do NOT add information from your training data. Never give legal advice.

{context}"""
        else:
            system_content = """You are HR Assist, a company HR policy assistant.
Answer based on the conversation history. Be concise, friendly, and professional."""

        if eval_retries > 0:
            system_content += "\n\nIMPORTANT: Your previous answer did not meet quality standards. Answer using ONLY information explicitly stated in the context above."

        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user"
                           else AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))

        response    = llm.invoke(lc_msgs)
        answer_text = response.content
        updates     = {"answer": answer_text}

        # Detect if answer warrants HR escalation
        lower = answer_text.lower()
        if any(phrase in lower for phrase in ["contact hr", "raise a ticket", "speak to hr", "hr team", "icc", "disciplinary", "pip"]):
            updates["escalate_to_hr"] = True
        else:
            updates["escalate_to_hr"] = False

        return updates

    def eval_node(state: CapstoneState) -> dict:
        answer  = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""

        result = llm.invoke(prompt).content.strip()
        try:
            score = float(result.split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        return {"messages": messages + [{"role": "assistant", "content": state["answer"]}]}

    return (memory_node, router_node, retrieval_node, skip_retrieval_node,
            tool_node, answer_node, eval_node, save_node)


# ── Graph Assembly ─────────────────────────────────────────
def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"


def build_agent():
    """
    Build and return the compiled HR Assist agent, embedder, and ChromaDB collection.

    Returns:
        app        — compiled LangGraph app, ready for .invoke()
        embedder   — SentenceTransformer for encoding queries
        collection — ChromaDB collection with 12 HR policy KB documents
    """
    llm      = ChatGroq(model=MODEL_NAME, temperature=0)
    embedder = SentenceTransformer(EMBED_MODEL)

    # Build ChromaDB
    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb")
    except:
        pass
    collection = client.create_collection("capstone_kb")

    texts      = [d["text"]  for d in DOCUMENTS]
    ids        = [d["id"]    for d in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
    )

    # Build nodes
    (memory_node, router_node, retrieval_node, skip_retrieval_node,
     tool_node, answer_node, eval_node, save_node) = make_nodes(llm, embedder, collection)

    # Assemble graph
    graph = StateGraph(CapstoneState)
    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"}
    )
    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())
    print(f"✅ HR Assist agent ready — {collection.count()} documents loaded")
    return app, embedder, collection