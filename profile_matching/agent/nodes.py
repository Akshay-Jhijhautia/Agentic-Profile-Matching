"""Graph nodes. Each takes the state and returns only the keys it changes."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.types import Send, interrupt

from .. import changes, explainability, llm, pools, prompts, reports, screening
from ..rag import resume_rag
from ..requirements import extract_requirements, load_job_description
from ..schemas import ChangeNarrative, RankingDiff, Requirements
from ..services import Services
from .state import AgentState
from .text import last_ai_text, last_human_text

# Messages of history the assistant sees, cut at a user message so tool calls
# and their results stay together.
ASSISTANT_HISTORY = 40


class PipelineNodes:
    """The screening pipeline, bound to one ``Services``."""

    def __init__(self, services: Services):
        self.services = services

    def parse_jd(self, state: AgentState) -> dict:
        """Load the job description: a JD file the router found in the message, otherwise
        the latest user message itself (text or a JD file path)."""
        text = load_job_description(state.get("jd_file") or last_human_text(state)).strip()
        if not text:
            raise ValueError("No job description given")
        return {"job_description": text, "jd_file": None}

    def extract_requirements(self, state: AgentState) -> dict:
        """Split the JD into must-have and nice-to-have requirements."""
        requirements = extract_requirements(state["job_description"], model=self.services.model)
        # A new search starts a new ranking: earlier refinement changes no longer apply.
        return {"requirements": requirements, "requirements_history": [requirements], "ranking_diff": None}

    def refine_requirements(self, state: AgentState) -> dict:
        """Apply the user's change to the current requirements, as a new version."""
        current = state["requirements"]
        instruction = f"Current requirements:\n{current.model_dump_json()}\n\nInstruction: {last_human_text(state)}"
        updated = llm.structured(Requirements, prompts.REFINE_REQUIREMENTS, instruction, model=self.services.model)
        return {"requirements": updated, "requirements_history": [updated]}

    def resumes_loaded(self, state: AgentState) -> str:
        """Conditional edge after the requirements: screen, or ask for resumes first."""
        return "search_resumes" if self.services.pool is not None else "await_resumes"

    def await_resumes(self, state: AgentState) -> dict:
        """Requirements are set but no resumes are loaded: say what to do next."""
        lines = reports.requirements_lines(state["requirements"])
        return {"messages": [AIMessage(
            f"Got it, screening for **{state['requirements'].role_title}**:\n\n"
            + "\n".join(f"- {line}" for line in lines)
            + "\n\nNo resumes are loaded yet. Give me a folder with `/resumes <folder>` or files with "
            "`/add <file> ...` (you can drag files into the terminal), and I'll screen them."
        )]}

    def search_resumes(self, state: AgentState) -> dict:
        """Round 1: score and check every indexed resume against the requirements."""
        return {
            "previous_candidates": state.get("candidates", []),
            "candidates": screening.evaluate(self.services, state["requirements"]),
        }

    def rank_candidates(self, state: AgentState) -> dict:
        """Round 1 cut: keep the best candidates who pass every must-have."""
        size = self.services.settings.shortlist_size
        return {
            "previous_shortlist": state.get("shortlist", []),
            "shortlist": screening.shortlist(state["candidates"], size),
        }

    def fan_out_assessments(self, state: AgentState) -> list[Send] | str:
        """Conditional edge: one parallel ``assess_candidate`` per shortlisted candidate
        without a cached assessment, or straight on when there is nothing to assess."""
        if self.services.settings.screening_rounds < 2:
            return "finalize_ranking"
        requirements, cached = state["requirements"], state.get("assessments", {})
        tasks = [
            Send("assess_candidate", {"requirements": requirements, "match": match})
            for match in state["shortlist"]
            if screening.assessment_key(self.services.pool.id, requirements, match.id) not in cached
        ]
        return tasks or "finalize_ranking"

    def assess_candidate(self, task: dict) -> dict:
        """Round 2 for one candidate (runs in parallel with the others)."""
        requirements, match = task["requirements"], task["match"]
        assessment = screening.assess(self.services, requirements, match)
        return {"assessments": {screening.assessment_key(self.services.pool.id, requirements, match.id): assessment}}

    def finalize_ranking(self, state: AgentState) -> dict:
        """Re-rank the shortlist by final score (round 2), then decide hire or no-hire (round 3)."""
        rounds, requirements = self.services.settings.screening_rounds, state["requirements"]
        shortlist = state["shortlist"]
        if rounds >= 2:
            shortlist = screening.apply_assessments(
                self.services.pool.id, shortlist, requirements, state.get("assessments", {})
            )
        if rounds >= 3:
            shortlist = screening.decide(self.services, requirements, shortlist)
        return {"shortlist": shortlist}

    def suggest_improvements(self, state: AgentState) -> dict:
        """Borderline candidates: what would turn them into a clear hire (needs round 2)."""
        if not any(match.borderline for match in state["shortlist"]):
            return {}
        return {"shortlist": screening.suggest_improvements(self.services, state["requirements"], state["shortlist"])}

    def explain_changes(self, state: AgentState) -> dict:
        """After a refinement: what changed in the requirements and the ranking, and why."""
        history = state["requirements_history"]
        diff = changes.ranking_diff(
            history[-2], history[-1],
            state["previous_candidates"], state["candidates"],
            state["previous_shortlist"], state["shortlist"],
        )
        return {"ranking_diff": diff}

    def generate_report(self, state: AgentState) -> dict:
        """Write the shortlist report and a short summary for the conversation."""
        parts = (state["requirements"], state["candidates"], state["shortlist"])
        diff = state.get("ranking_diff")
        report = reports.shortlist_report(*parts, diff=diff)
        summary = reports.shortlist_summary(*parts)
        if diff is not None:
            summary = f"{self._narrate(diff)}\n\n{summary}"
        report_dir = None
        if self.services.settings.save_reports:
            folder = explainability.save_reports(
                self.services.settings.reports_dir, state["requirements"], state["shortlist"], report
            )
            report_dir = resume_rag.display_path(str(folder))
            summary += f"\n\nReports saved to `{report_dir}`."
        return {"report": report, "report_dir": report_dir, "messages": [AIMessage(summary)]}

    def _narrate(self, diff: RankingDiff) -> str:
        if not diff.requirement_changes:
            return "The requirements did not change, so neither did the ranking."
        moved = diff.of_kind("new", "dropped", "up", "down")
        payload = diff.model_copy(update={"changes": moved}).model_dump_json()
        return llm.structured(ChangeNarrative, prompts.EXPLAIN_CHANGES, payload, model=self.services.model).narrative


class ConversationNodes:
    """The human feedback loop and the tool-using assistant."""

    def __init__(self, services: Services, tools: list[BaseTool]):
        self.services = services
        self.tools = tools

    def load_resumes(self, state: AgentState) -> dict:
        """Activate a folder, or upload files into this conversation's batch; earlier
        results belonged to the previous resumes."""
        batch = state.get("upload_folder") or pools.new_batch(self.services.settings.uploads_dir)
        try:
            report, uploaded_to = self.services.use_paths(state["load_paths"], batch=batch)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            return {"load_paths": None, "load_error": str(exc),
                    "messages": [AIMessage(f"I couldn't load those resumes: {exc}")]}
        reply = report.summary()
        requirements = state.get("requirements")
        if state["intent"] != "new_search":
            reply += (f"\n\nScreening them for {requirements.role_title}…" if requirements
                      else "\n\nDescribe the role or give a job description to screen them.")
        return {
            "load_paths": None, "load_error": None, "resume_folder": report.folder,
            "pool_folder": str(self.services.pool.folder),
            "upload_folder": str(uploaded_to) if uploaded_to else state.get("upload_folder"),
            "candidates": [], "shortlist": [], "previous_candidates": [], "previous_shortlist": [],
            "ranking_diff": None, "messages": [AIMessage(reply)],
        }

    @staticmethod
    def human_feedback(state: AgentState) -> dict:
        """Pause the graph until the user replies (resumed with ``Command(resume=text)``)."""
        reply = interrupt({"last_message": last_ai_text(state)})
        return {"messages": [HumanMessage(reply)]}

    def assistant(self, state: AgentState) -> dict:
        """Answer the user, calling tools as needed (loops with the ``tools`` node)."""
        context = reports.conversation_context(
            state.get("requirements"), state.get("candidates", []), state.get("shortlist", []),
            state.get("ranking_diff"),
        )
        pool = self.services.pool
        source = state.get("resume_folder") or (resume_rag.display_path(str(pool.folder)) if pool else None)
        loaded = (f"Resumes: {source} ({pool.size} resumes)." if pool
                  else "No resumes are loaded yet.")
        context = f"{loaded} The user loads resumes with /resumes <folder> or /add <files>.\n{context}"
        history = trim_messages(
            state["messages"], max_tokens=ASSISTANT_HISTORY, token_counter=len, start_on="human"
        )
        model = llm.with_tools(self.tools, model=self.services.model)
        reply = model.invoke([SystemMessage(prompts.ASSISTANT.format(context=context)), *history])
        return {"messages": [reply]}
