# agents.py
import re
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama

# 1. Initialize shared LLM (Llama3 via Ollama)
llm = Ollama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    request_timeout=600.0,    # optional: increase timeout
)

# 2. Dummy data
DUMMY_RESUMES = {
    "c1": {"name":"Alice","skills":["Python","ML"],"experience":"5y","email":"alice@example.com"}
}
DUMMY_JOBS = {
    "j1": {
        "title": "Senior Data Scientist",
        "required_skills": ["Python", "Machine Learning", "Data Analysis"],
        "company": "Tech Innovators",
        "recruiter_email": "hr@techinnovators.com"
    },
    "j2": {
        "title": "Frontend Developer",
        "required_skills": ["JavaScript", "React", "CSS"],
        "company": "Web Solutions",
        "recruiter_email": "jobs@websolutions.com"
    }
}
DUMMY_SLOTS = ["2025-08-01T10:00","2025-08-01T14:00"]

# 3. Sub-agents
resume_parser_agent = FunctionAgent(
    name="ResumeParserAgent",
    description="Parse candidate resume into structured profile",
    system_prompt="""
You are a resume parsing specialist. Given candidate data, extract skills, experience, and contact info.""",
    llm=llm,
    tools=[],
    initial_state={"parsed": {}}
)

job_matcher_agent = FunctionAgent(
    name="JobMatcherAgent",
    description="Match candidate profile to jobs",
    system_prompt="""
You are a job matching specialist. Given a profile and job list, return top matches.""",
    llm=llm,
    tools=[],
    initial_state={"matches": {}}
)

application_agent = FunctionAgent(
    name="ApplicationAgent",
    description="Handle application submission and notifications",
    system_prompt="""
You are an application manager. Record applications and notify recruiters.""",
    llm=llm,
    tools=[],
    initial_state={"applications": {}}
)

interview_scheduler_agent = FunctionAgent(
    name="InterviewSchedulerAgent",
    description="Present interview slots to candidate",
    system_prompt="""
You are an interview scheduler. Given recruiter availability, list slots.""",
    llm=llm,
    tools=[],
    initial_state={"interviews": {}}
)

calendar_agent = FunctionAgent(
    name="CalendarAgent",
    description="Create calendar events for confirmed interviews",
    system_prompt="""
You are a calendar manager. Create an event for the selected slot and return confirmation.""",
    llm=llm,
    tools=[],
    initial_state={"events": {}}
)

# 4. Tool wrappers
async def call_parse(ctx: Context, candidate_id: str) -> str:
    data = DUMMY_RESUMES.get(candidate_id)
    if not data: return "Candidate not found"
    res = await resume_parser_agent.run(
        user_msg=f"Parse this resume: {data}"
    )
    state = await ctx.store.get("state")
    state["candidates"][candidate_id] = data
    await ctx.store.set("state", state)
    return str(res)

async def call_match(ctx: Context, candidate_id: str) -> str:
    profile = (await ctx.store.get("state"))["candidates"].get(candidate_id)
    if not profile: return "Parse resume first"
    res = await job_matcher_agent.run(
        user_msg=f"Match profile {profile} against jobs {DUMMY_JOBS}"
    )
    # simplistic match
    matches = [jid for jid, j in DUMMY_JOBS.items()
               if any(s in profile["skills"] for s in j["required_skills"])]
    state = await ctx.store.get("state")
    state["candidates"][candidate_id]["matches"] = matches
    await ctx.store.set("state", state)
    return str(res)

async def call_apply(ctx: Context, candidate_id: str, job_id: str) -> str:
    res = await application_agent.run(
        user_msg=f"Candidate {candidate_id} applies to job {job_id}"
    )
    app_id = f"{candidate_id}_{job_id}"
    state = await ctx.store.get("state")
    state["applications"][app_id] = {"candidate_id":candidate_id,"job_id":job_id}
    await ctx.store.set("state", state)
    return str(res)

async def call_slots(ctx: Context, application_id: str) -> str:
    res = await interview_scheduler_agent.run(
        user_msg=f"Show slots for application {application_id}"
    )
    state = await ctx.store.get("state")
    state["interviews"][application_id] = {"slots": DUMMY_SLOTS}
    await ctx.store.set("state", state)
    return str(res)

async def call_event(ctx: Context, application_id: str, slot: str) -> str:
    res = await calendar_agent.run(
        user_msg=f"Create event for {application_id} at {slot}"
    )
    state = await ctx.store.get("state")
    iv = state["interviews"][application_id]
    iv.update({"confirmed_slot": slot, "event_id": f"evt_{application_id}"})
    await ctx.store.set("state", state)
    return str(res)

# 5. Convert to FunctionTool
parse_tool       = FunctionTool.from_defaults(call_parse)
match_tool       = FunctionTool.from_defaults(call_match)
apply_tool       = FunctionTool.from_defaults(call_apply)
slots_tool       = FunctionTool.from_defaults(call_slots)
calendar_tool    = FunctionTool.from_defaults(call_event)

# 6. Orchestrator Agent
orchestrator = FunctionAgent(
    name="RecruitmentOrchestrator",
    description="Orchestrate resume → match → apply → schedule → calendar",
    system_prompt="""
You are the recruitment orchestrator. Use tools to:
1. parse resume
2. match jobs
3. handle application
4. present slots
5. create calendar event
Follow this exact sequence when asked.""",
    llm=llm,
    tools=[parse_tool, match_tool, apply_tool, slots_tool, calendar_tool],
    verbose=True,
    initial_state={"candidates":{}, "applications":{}, "interviews":{}}
)
