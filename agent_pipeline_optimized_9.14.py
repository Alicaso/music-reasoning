#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized multi-turn music reasoning agent (planner → thinker/tool-user loop → reviewer)

Key optimizations:
1. Unified error return format: {"ok": true/false, "data": "..."}
2. Structured parameter validation: Using JSON Schema to validate tool parameters
3. Optimized information flow: thinker only gets tool names and descriptions, tooler gets detailed info for selected tools
4. Enhanced error handling: parameter type checking, required parameter validation, enum value validation
5. Improved prompts: clearer agent role definitions and task division

Environment requirements:
- Requires OPENAI_API_KEY environment variable
- Dependencies: openai>=1.0.0, pyyaml, jsonschema
- Requires music21_tools.py and music21_tools.yaml files

Usage:
    python agent_pipeline_optimized.py --question_id 001
"""

import argparse
import csv
import importlib
import inspect
import json
import os
import re
import multiprocessing as mp
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
import jsonschema

# --- Multiprocessing worker (module-level, spawn-safe) ---
def _mp_tool_worker(q: "mp.Queue", module_name: str, function_name: str, kwargs: Dict[str, Any]) -> None:
    """Import the module and call the named function with kwargs, returning result via Queue.
    This must be module-level to be picklable on 'spawn' start method (macOS).
    """
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, function_name)
        res = fn(**kwargs)
        q.put((True, res, None))
    except Exception as e:
        try:
            q.put((False, None, str(e)))
        except Exception:
            # As a last resort, avoid crashing the child
            pass

# --- Config ---
CSV_PATH = os.environ.get("QUESTIONS_CSV_PATH", "test.csv")
KERN_FOLDER = os.environ.get("KERN_FOLDER", "kern_exam_new")
TOOLS_MODULE = os.environ.get("TOOLS_MODULE", "music21_tools")
TOOLS_YAML = os.environ.get("TOOLS_YAML", "music21_tools.yaml")

OPENAI_MODEL_PLANNER = os.environ.get("PLANNER_MODEL", "gpt-4.1-mini")
OPENAI_MODEL_THINKER = os.environ.get("THINKER_MODEL", "gpt-4.1-mini")
OPENAI_MODEL_TOOLER = os.environ.get("TOOLER_MODEL", "gpt-4.1-mini")
OPENAI_MODEL_REVIEWER = os.environ.get("REVIEWER_MODEL", "gpt-4.1-mini")

MAX_STEPS = 12
TOOL_RETRY = 3

# --- OpenAI client ---
try:
    # Prefer SDK v1 interface
    from openai import OpenAI  # type: ignore
    _openai_client = OpenAI()
    _OPENAI_TIMEOUT = float(os.environ.get("OPENAI_TIMEOUT", "60"))
    _OPENAI_MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))
    _OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0"))
    def ask_openai(model: str, system: str, user: str) -> str:
        client = _openai_client.with_options(timeout=_OPENAI_TIMEOUT, max_retries=_OPENAI_MAX_RETRIES)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=_OPENAI_TEMPERATURE,
        )
        return resp.choices[0].message.content or ""
except ImportError:
    # Fallback for legacy SDK (<1.0.0) if installed
    import openai  # type: ignore
    _OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0"))
    def ask_openai(model: str, system: str, user: str) -> str:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=_OPENAI_TEMPERATURE,
        )
        return resp["choices"][0]["message"]["content"]

# --- Data structures ---
@dataclass
class Turn:
    role: str  # 'planner' | 'thinker' | 'tooler' | 'tool' | 'reviewer' | 'system'
    content: str
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')

@dataclass
class EpisodeState:
    question_id: str
    music_id: str
    question: str
    answer_gt: str  # ground-truth letter a/b/c/d
    kern_text: str
    messages: List[Turn] = field(default_factory=list)
    recent_tool_calls: List[str] = field(default_factory=list)
    # Detailed trace of each round/phase input & output & results
    trace_events: List[Dict[str, Any]] = field(default_factory=list)
    # Persistent plan to include in every thinker prompt
    plan: str = ""

@dataclass
class ToolCallResult:
    success: bool
    data: str
    error: Optional[str] = None

# --- Utilities ---

def read_csv_row_by_qid(csv_path: str, qid: str) -> Tuple[str, str, str, str]:
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get('question_id', '')).strip() == str(qid).strip():
                # Try both 'music_id' and 'question_id' columns for kern file name
                music_id = str(row.get('music_id', '')).strip()
                if not music_id:
                    music_id = str(row.get('question_id', '')).strip()
                question = str(row.get('question', '')).strip()
                # Try both 'answer' and 'truth_letter' columns
                answer = str(row.get('answer', '')).strip().upper()
                if not answer or answer not in {"A", "B", "C", "D"}:
                    answer = str(row.get('truth_letter', '')).strip().upper()
                if answer not in {"A", "B", "C", "D"}:
                    raise ValueError(f"Row {qid}: answer must be A/B/C/D, got: {answer}")
                return music_id, question, answer, json.dumps(row, ensure_ascii=False)
    raise KeyError(f"question_id {qid} not found in {csv_path}")

def load_kern_text(music_id: str, folder: str = KERN_FOLDER) -> str:
    # Try different filename formats
    possible_paths = [
        os.path.join(folder, f"{music_id}.krn"),
        os.path.join(folder, f"{music_id.zfill(3)}.krn"),  # Pad with zeros
        os.path.join(folder, f"{music_id}.txt"),
        os.path.join(folder, f"{music_id.zfill(3)}.txt"),  # Pad with zeros
    ]
    
    for fp in possible_paths:
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                return f.read()
    
    raise FileNotFoundError(f"KERN file for music_id={music_id} not found. Tried: {possible_paths}")

def load_tools_module(module_name: str = TOOLS_MODULE):
    try:
        if module_name == "music21_tools":
            # Use the wrapped tool module
            import tool_wrapper
            return tool_wrapper
        else:
            return importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Failed to import tools module '{module_name}': {e}")

def load_tools_yaml(yaml_path: str = TOOLS_YAML) -> Tuple[str, Dict[str, Any]]:
    if not os.path.exists(yaml_path):
        return "", {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        data = yaml.safe_load(text) or {}
    except Exception:
        data = {}
    return text, data

def list_module_functions(mod) -> List[Tuple[str, str]]:
    """Return list of (name, signature_str) for callables defined in the module."""
    funcs = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        # Accept functions exposed on the module, even if their __module__ differs
        # (e.g., wrappers defined in tool_wrapper that preserve original __module__)
        if inspect.isfunction(obj):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(…)"
            funcs.append((name, sig))
    return funcs

def ensure_json(text: str) -> Dict[str, Any]:
    """Best-effort to extract a JSON object from LLM output."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try fenced JSON code blocks
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try first { ... } block
    m = re.search(r"(\{[\s\S]*?\})", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Try to fix common JSON issues: unquoted variable names
    try:
        # Fix unquoted variable names like "kern_data": kern_text -> "kern_data": "kern_text"
        fixed_text = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2', text)
        return json.loads(fixed_text)
    except Exception:
            pass
    raise ValueError("Failed to parse JSON from LLM output:\n" + text[:800])

def log_event(state: EpisodeState, round_no: int, phase: str, event_type: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """Append a structured trace event to the episode state.
    - round_no: 0 for planner, 1..N for thinker/tooler/tool, -1 for reviewer
    - phase: planner | thinker | tooler | tool | reviewer | system
    - event_type: input/system | input/user | output/model | call | result | error | info
    - content: main textual payload
    - meta: optional JSON-serializable dict with extra fields
    """
    try:
        event_obj: Dict[str, Any] = {
            "round": round_no,
            "phase": phase,
            "type": event_type,
            "content": content,
        }
        if meta:
            event_obj["meta"] = meta
        state.trace_events.append(event_obj)
    except Exception:
        # Never break pipeline due to logging issues
        pass

def validate_tool_arguments(tool_schema: Dict[str, Any], arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate tool arguments against JSON schema."""
    try:
        jsonschema.validate(instance=arguments, schema=tool_schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, f"Parameter validation failed: {e.message}"
    except Exception as e:
        return False, f"Schema validation error: {str(e)}"

def execute_tool_with_validation(tools_mod, tool_name: str, arguments: Dict[str, Any], 
                                tool_schema: Dict[str, Any]) -> ToolCallResult:
    """Execute tool with parameter validation and standardized error handling."""
    
    # 1. Validate arguments
    is_valid, error_msg = validate_tool_arguments(tool_schema, arguments)
    if not is_valid:
        return ToolCallResult(success=False, data="", error=error_msg)
    
    # 2. Get function and execute (with timeout guard)
    try:
        # Ensure the function exists on the provided tools module
        _ = getattr(tools_mod, tool_name)
        tool_timeout_s = float(os.environ.get("TOOL_EXEC_TIMEOUT", "45"))

        ctx = mp.get_context(os.environ.get("MP_START_METHOD", "spawn"))
        q: mp.Queue = ctx.Queue()
        p = ctx.Process(target=_mp_tool_worker, args=(q, tools_mod.__name__, tool_name, arguments))
        p.start()
        p.join(tool_timeout_s)
        if p.is_alive():
            try:
                p.terminate()
            finally:
                p.join(1)
            return ToolCallResult(success=False, data="", error=f"Timeout after {tool_timeout_s:.0f}s")

        ok, res, err = q.get() if not q.empty() else (False, None, "No result returned")
        if not ok:
            return ToolCallResult(success=False, data="", error=f"Execution error: {err}")

        result = res
        # 3. Result is already in standardized format from tool_wrapper
        if isinstance(result, dict) and "ok" in result and "data" in result:
            if result["ok"]:
                return ToolCallResult(success=True, data=result["data"])
            else:
                return ToolCallResult(success=False, data="", error=result["data"])
        else:
            # Fallback for unexpected format
            result_str = str(result)
            return ToolCallResult(success=True, data=result_str)
            
    except AttributeError:
        # Fallback: try calling function directly from raw module if wrapper does not expose it
        try:
            raw_mod = importlib.import_module("music21_tools")
            raw_func = getattr(raw_mod, tool_name)
        except Exception:
            return ToolCallResult(success=False, data="", error=f"Tool function '{tool_name}' does not exist")

        # Execute raw function with the same timeout protection
        try:
            tool_timeout_s = float(os.environ.get("TOOL_EXEC_TIMEOUT", "45"))

            ctx = mp.get_context(os.environ.get("MP_START_METHOD", "spawn"))
            qf: mp.Queue = ctx.Queue()
            pf = ctx.Process(target=_mp_tool_worker, args=(qf, "music21_tools", tool_name, arguments))
            pf.start()
            pf.join(tool_timeout_s)
            if pf.is_alive():
                try:
                    pf.terminate()
                finally:
                    pf.join(1)
                return ToolCallResult(success=False, data="", error=f"Timeout after {tool_timeout_s:.0f}s")

            okf, resf, errf = qf.get() if not qf.empty() else (False, None, "No result returned")
            if not okf:
                return ToolCallResult(success=False, data="", error=f"Execution error: {errf}")

            if isinstance(resf, dict) and "ok" in resf and "data" in resf:
                if resf["ok"]:
                    return ToolCallResult(success=True, data=resf["data"])
                else:
                    return ToolCallResult(success=False, data="", error=resf["data"])
            else:
                return ToolCallResult(success=True, data=str(resf))
        except Exception as e2:
            return ToolCallResult(success=False, data="", error=f"Execution error: {str(e2)}")
    except Exception as e:
        return ToolCallResult(success=False, data="", error=f"Execution error: {str(e)}")

def get_tool_schema(tools_yaml: Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
    """Get the schema for a specific tool."""
    for tool in tools_yaml:
        if tool.get('tool_name') == tool_name:
            return tool.get('args_schema')
    return None

def get_tool_info_for_thinker(tools_yaml: Dict[str, Any]) -> str:
    """Get simplified tool information for thinker agent."""
    tool_descriptions = []
    for tool in tools_yaml:
        name = tool.get('tool_name', '')
        brief_desc = tool.get('brief_description', tool.get('description', ''))
        tool_descriptions.append(f"- {name}: {brief_desc}")
    return "\n".join(tool_descriptions)

def get_tool_info_for_tooler(tools_yaml: Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed tool information for tooler agent."""
    for tool in tools_yaml:
        if tool.get('tool_name') == tool_name:
            return {
                'name': tool.get('tool_name'),
                'description': tool.get('description'),
                'arguments': tool.get('arguments', []),
                'args_schema': tool.get('args_schema', {}),
                'examples': tool.get('examples', [])
            }
    return None

# --- Agents (Prompts) ---
PLANNER_SYS = (
    "You are a methodical Music Analyst and strategic Planner. Your primary goal is to create a logical, step-by-step plan for answering a multiple-choice musical question.\n\n"
    "**Core Principles You Must Follow:**\n"
    "1. First, write a short paragraph under the field `thought`. This should be a natural-language explanation of why you designed your plan. Keep it concise (not extremely long), but it should be substantive enough to justify the plan. Explicitly mention how you will interact with the provided answer choices (i.e., what evidence you will compare against each option).\n"
    "2. Next, write a clear sequence of 3–8 steps under the field `plan`. Each step should state what to analyze or compare (e.g., pitch, rhythm, harmony, form) without naming specific algorithms or implementation details. Include at least one step that explicitly says you will compare your findings to the given options and select the best match.\n"
    "3. If relevant, account for a possible pickup (incomplete first measure) when interpreting measure references. In this case, the measure number mentioned in the question could either refer to the count that includes the pickup measure, or to the count that excludes it. \n"
    "4. Do not perform the actual analysis yourself. Only output the reasoning `thought` and the `plan`.\n\n"

    "**Output Format (STRICT JSON):**\n"
    "{\n"
    "  \"thought\": \"A short natural-language explanation of why you designed the plan this way, including how you will compare against the options.\",\n"
    "  \"plan\": [\n"
    "    \"Step 1 ...\",\n"
    "    \"Step 2 ...\"\n"
    "  ]\n"
    "}\n\n"

    "**Here are two output examples:**\n"
    "Example 1 (Detect repeated sections and name the form):\n"
    "{\n"
    "  \"thought\": \"To determine whether the excerpt contains repeated sections and how that maps to a named form, I should first convert the score into a simplified representation: a melodic line (relative intervals/contour) and a rhythmic skeleton. With these simplified lines, I can scan for recurring or transposed fragments and only then infer cut points for segments. After establishing likely segments and their recurrence order, I will compare the observed layout (e.g., AA, AB, AABA, or a varied reprise) against the answer choices, selecting the option whose description best matches the detected repetition pattern.\",\n"
    "  \"plan\": [\n"
    "    \"Derive simplified melodic and rhythmic lines from the kern data (e.g., contour and basic durations)\",\n"
    "    \"Search these simplified lines for recurring or transposed fragments to infer natural segmentation points\",\n"
    "    \"Label segments by their recurrence pattern (e.g., A, B, A') and outline the overall order\",\n"
    "    \"Map the observed pattern to common form labels (e.g., AA, AB, AABA)\",\n"
    "    \"Compare the resulting pattern and label to each provided option and choose the closest match\"\n"
    "  ]\n"
    "}\n\n"
    "Example 2 (Choose the best revision advice among options):\n"
    "{\n"
    "  \"thought\": \"Since the author reports that the piece feels 'odd' and wants practical fixes, I should first note what each option is proposing (e.g., clarify cadences/phrasing, introduce a steadier rhythmic foundation, adjust harmony for functional flow, or apply a stylistic texture change). I will then inspect the music at a high level—melody/phrase shape, cadence clarity, harmonic direction, and rhythmic grounding—to see which issue actually explains the ‘odd’ impression. Finally, I will compare these observations against each option’s remedy and select the option that addresses the real problem without introducing stylistic mismatches.\",\n"
    "  \"plan\": [\n"
    "    \"List the kinds of remedies the options propose (e.g., cadence/phrase clarification, steadier accompaniment pattern, more functional progressions, or a texture/orchestration change)\",\n"
    "    \"Check phrase structure and cadence points for clarity (do phrases close convincingly, or feel dangling?)\",\n"
    "    \"Assess harmonic flow for functional direction (are there weak or meandering progressions that undermine resolution?)\",\n"
    "    \"Evaluate rhythmic foundation for consistency and support (does an accompaniment pattern or ostinato help the melody?)\",\n"
    "    \"Compare these findings to each option’s proposed fix and select the advice that best addresses the diagnosed cause of the ‘odd’ feeling\"\n"
    "  ]\n"
    "}\n"
)

PLANNER_USER_TMPL = (
    "Now, here is the question and the kern score, please do your own thinking and output the plan:\n"
    "Question (Multiple Choice):\n{question}\n\n"
    "Full kern score:\n{kern}\n\n"
    "Remember to output strict JSON as instructed."
)

THINKER_SYS = (
    "You are a Thinker agent in a ReAct-style loop. Choose one of: THINK, CALL_TOOL, FINISH.\n\n"
    "Output strict JSON with: decision ('THINK'|'CALL_TOOL'|'FINISH'), thought (string).\n"
    "If current information is sufficient to decide the final answer, choose FINISH.\n\n"
    "**CRITICAL: Always read the results from previous tool calls and thought before proceeding.**\n"
    "When thinking, do some real reasoning instead of just planning for the next step.\n"
    "For global questions, consider the whole score instead of just a few measures.\n"
    "Call tools only when they will clearly outperform your own reasoning. Avoid using tools blindly or for analyses that are weakly related to the question.\n"
    "IMPORTANT: Before calling a tool, check if you have already called the same tool with the same parameters recently. Avoid duplicate tool calls with same parameters!\n\n"
    "SPECIAL NOTE: If you encounter unfamiliar music theory concepts, terms, or need clarification on specialized terminology, use the search_music_concept tool to get accurate definitions and explanations. This can be particularly helpful for genre identification, compositional techniques, or theoretical concepts you're unsure about.\n"
)

THINKER_USER_TMPL = (
    "Question (Multiple Choice):\n{question}\n\n"
    "Available tools (names and descriptions):\n{tool_descriptions}\n\n"
    "Initial context (plan + full kern):\n{initial_context}\n\n"
    "Recent context (last {ctx_n} messages):\n{recent_context}\n\n"
    "**IMPORTANT: You have a maximum of 15 rounds for thinking and reasoning. This is round {current_round}/15.**\n"
    "Please ensure you complete your reasoning before the limit.\n\n"
    "Please decide the next action.\n"
    "If CALL_TOOL, include: tool_name (string), usage_statement (string describing the intended goal and requirements).\n"
    "Example for CALL_TOOL: {{\"decision\": \"CALL_TOOL\", \"thought\": \"...\", \"tool_name\": \"get_chord_name_at_event\", \"usage_statement\": \"...\"}}\n"
    "Do NOT invent or list specific argument values, describe the possible arguments in natural language in 'usage_statement'."
)

TOOLER_SYS = (
    "You are a Tool-Using agent. Given the selected tool and the Thinker's usage_statement, produce the exact Python call spec.\n"
    "You will receive: (1) tool_name, (2) usage_statement, (3) function signature, (4) tool schema and examples.\n"
    "Available runtime variables: 'kern_text', 'question_text', 'music_id', 'question_id'.\n"
    "Output strict JSON: {\"function\": \"...\", \"arguments\": {...}, \"justification\": \"...\"}.\n"
    "Choose arguments and values yourself based on schema and intent. Do not return code.\n"
)

TOOLER_USER_TMPL = (
    "Selected tool name: {tool_name}\n"
    "Usage_statement (from thinker): {usage}\n\n"
    "Full kern text is available as runtime variable 'kern_text'.\n\n"
    "Tool schema & examples:\n{tool_info}\n\n"
    "Function signature:\n{func_sigs}\n\n"
    "Now return JSON call specification."
)

REVIEWER_SYS = (
    "You are the final Reviewer. Read the entire conversation (planner, thinker steps, tool outputs).\n"
    "Decide the best answer, choose from options A/B/C/D.\n"
    "Only output a single uppercase letter: A or B or C or D. Do not explain.\n"
)

REVIEWER_USER_TMPL = (
    "Question (Multiple Choice):\n{question}\n\n"
    "Conversation log:\n{transcript}\n\n"
)


# --- Core pipeline ---

def run_episode(qid: str, csv_path: str = None) -> Tuple[str, EpisodeState]:
    # Load inputs
    if csv_path is None:
        csv_path = CSV_PATH
    music_id, question_text, answer_gt, raw_data = read_csv_row_by_qid(csv_path, qid)
    kern_text = load_kern_text(music_id)
    
    # For reddit dataset, format question with options
    import json
    import ast
    try:
        raw_dict = json.loads(raw_data)
        if 'final_options' in raw_dict:
            # final_options is a string like "['xxx','xxx','xxx','xxx']", parse it as Python literal
            options_str = raw_dict['final_options']
            options = ast.literal_eval(options_str)  # Parse string representation of list
            formatted_question = question_text + "\n"
            for i, option in enumerate(options):
                letter = chr(ord('A') + i)  # A, B, C, D
                formatted_question += f"{letter}. {option}\n"
            question_text = formatted_question
    except (json.JSONDecodeError, KeyError, ValueError, SyntaxError) as e:
        # If parsing fails, use original question
        print(f"Warning: Failed to parse options for {qid}: {e}")
        pass

    state = EpisodeState(
        question_id=qid,
        music_id=music_id,
        question=question_text,
        answer_gt=answer_gt,
        kern_text=kern_text,
    )

    # Load tools
    tools_mod = load_tools_module(TOOLS_MODULE)
    tools_yaml_text, tools_yaml = load_tools_yaml(TOOLS_YAML)
    func_list = list_module_functions(tools_mod)

    # 1) Planner
    planner_user_msg = PLANNER_USER_TMPL.format(
        qid=qid,
        mid=music_id,
        question=question_text,
        kern=kern_text,
    )
    log_event(state, 0, "planner", "input/system", PLANNER_SYS)
    log_event(state, 0, "planner", "input/user", planner_user_msg)
    plan = ask_openai(
        OPENAI_MODEL_PLANNER,
        PLANNER_SYS,
        planner_user_msg,
    )
    plan_augmented = (plan or "").strip()

    # Parse planner JSON to obtain plan array
    plan_json: Dict[str, Any]
    try:
        plan_json = ensure_json(plan)
    except Exception:
        # Fallback: wrap raw text into plan array
        plan_json = {"plan": [ (plan or "").strip() ]}

    plan_list = plan_json.get("plan") if isinstance(plan_json.get("plan"), list) else [str(plan_json.get("plan", "")).strip()]
    # Note: thought field is ignored as it's only for planner's internal reasoning

    plan_text = "\n- ".join([s for s in (str(x).strip() for x in plan_list) if s])
    plan_text = ("- " + plan_text) if plan_text else "(no plan provided)"

    plan_with_summary = f"Plan:\n{plan_text}"

    log_event(state, 0, "planner", "output/model", json.dumps(plan_json, ensure_ascii=False))
    state.messages.append(Turn("planner", plan_with_summary))

    # --- Store plan for thinker ---
    state.plan = plan_text
    log_event(state, 0, "planner", "info", "Plan prepared")

    # 2) Multi-turn loop
    for step in range(1, MAX_STEPS + 1):
        # Compose Thinker input
        recent_context_snippets: List[str] = []
        if step > 1:
            for t in state.messages:
                recent_context_snippets.append(f"[{t.role}] {t.content}")
        
        # Get simplified tool descriptions for thinker
        tool_descriptions = get_tool_info_for_thinker(tools_yaml)
        
        # Create initial context with plan and kern data
        initial_context = f"Plan:\n{state.plan}\n\nKern data:\n{kern_text}"
        
        thinker_user_msg = THINKER_USER_TMPL.format(
            qid=qid,
            mid=music_id,
            question=question_text,
            tool_descriptions=tool_descriptions,
            initial_context=initial_context,
            ctx_n=len(recent_context_snippets),
            recent_context="\n".join(recent_context_snippets) if recent_context_snippets else "(none)",
            current_round=step,
        )
        log_event(state, step, "thinker", "input/system", THINKER_SYS)
        log_event(state, step, "thinker", "input/user", thinker_user_msg)
        thinker_out_raw = ask_openai(
            OPENAI_MODEL_THINKER,
            THINKER_SYS,
            thinker_user_msg,
        )
        log_event(state, step, "thinker", "output/model", thinker_out_raw.strip())
        
        try:
            thinker_json = ensure_json(thinker_out_raw)
            log_event(state, step, "thinker", "info", "Parsed THINKER JSON", {"json": thinker_json})
        except Exception as e:
            state.messages.append(Turn("thinker", f"(Parse error) {thinker_out_raw.strip()}"))
            log_event(state, step, "thinker", "error", f"JSON parse error: {e}")
            thinker_json = {"decision": "THINK", "thought": "Due to JSON parse error, continuing with small reasoning step."}

        decision = str(thinker_json.get("decision", "")).strip().upper()
        thought = str(thinker_json.get("thought", "")).strip()

        # Always record thinker's thought first
        state.messages.append(Turn("thinker", f"{decision}: {thought}"))
        log_event(state, step, "thinker", "info", f"Decision {decision}: {thought}")

        if decision == "FINISH":
            break

        if decision == "THINK" or decision == "":
            continue

        if decision == "CALL_TOOL":
            tool_name = str(thinker_json.get("tool_name", "")).strip()
            # Prefer 'usage_statement'; fallback to 'usage_intent' and then 'tool_input_description'
            usage_raw = thinker_json.get("usage_statement")
            if usage_raw is None or (isinstance(usage_raw, str) and not usage_raw.strip()):
                usage_raw = thinker_json.get("usage_intent")
            if usage_raw is None or (isinstance(usage_raw, str) and not usage_raw.strip()):
                usage_raw = thinker_json.get("tool_input_description", "")
            usage = str(usage_raw or "").strip()
            
            # If no tool_name provided, try to extract from thought
            if not tool_name:
                # Try to extract tool name from thought using common patterns
                thought_text = str(thinker_json.get("thought", "")).lower()
                tool_candidates = []
                for tool in tools_yaml:
                    tool_name_candidate = tool.get('tool_name', '')
                    if tool_name_candidate and tool_name_candidate.lower() in thought_text:
                        tool_candidates.append(tool_name_candidate)
                
                if tool_candidates:
                    # Use the first matching tool
                    tool_name = tool_candidates[0]
                    log_event(state, step, "thinker", "info", f"Extracted tool_name '{tool_name}' from thought")
                else:
                    log_event(state, step, "thinker", "error", "CALL_TOOL requested but no tool_name provided")
                    continue

            # Get detailed tool info for tooler
            tool_info = get_tool_info_for_tooler(tools_yaml, tool_name)
            if not tool_info:
                log_event(state, step, "thinker", "error", f"Tool '{tool_name}' not found in tools_yaml")
                continue

            # Build tooler input
            func_sigs = "\n".join([f"- {n}{s}" for n, s in func_list if n == tool_name])
            # Slim tool info: remove verbose 'arguments' duplication; keep schema and optional examples
            tool_info_str = (
                f"Name: {tool_info['name']}\n"
                f"Description: {tool_info['description']}\n"
                f"Schema: {json.dumps(tool_info.get('args_schema', {}), ensure_ascii=False, indent=2)}\n"
                f"Examples: {json.dumps(tool_info.get('examples', []), ensure_ascii=False, indent=2)}"
            )

            tooler_user_msg = TOOLER_USER_TMPL.format(
                tool_name=tool_name,
                usage=(usage),
                tool_info=tool_info_str,
                func_sigs=func_sigs,
            )

            # Retry loop for tool generation/execution
            last_err: Optional[str] = None
            for attempt in range(1, TOOL_RETRY + 1):
                # If there was a previous error, append it to the tooler input
                effective_tooler_user_msg = TOOLER_USER_TMPL.format(
                    tool_name=tool_name,
                    usage=(usage + (f"\n(Previous error: {last_err})" if last_err else "")),
                    tool_info=tool_info_str,
                    func_sigs=func_sigs,
                )
                log_event(state, step, "tooler", "input/system", TOOLER_SYS, {"attempt": attempt})
                log_event(state, step, "tooler", "input/user", effective_tooler_user_msg, {"attempt": attempt})
                tooler_out_raw = ask_openai(
                    OPENAI_MODEL_TOOLER,
                    TOOLER_SYS,
                    effective_tooler_user_msg,
                )
                log_event(state, step, "tooler", "output/model", tooler_out_raw.strip(), {"attempt": attempt})
                
                try:
                    call_spec = ensure_json(tooler_out_raw)
                    log_event(state, step, "tooler", "info", "Parsed TOOLER JSON", {"attempt": attempt, "json": call_spec})
                except Exception as e:
                    last_err = f"Tooler JSON parse error: {e}"
                    state.messages.append(Turn("tooler", f"Attempt {attempt}: {last_err}"))
                    log_event(state, step, "tooler", "error", last_err, {"attempt": attempt})
                    continue

                fn = str(call_spec.get("function", "")).strip()
                args = call_spec.get("arguments", {})
                if not fn or not isinstance(args, dict):
                    last_err = "Invalid call specification: missing function name or arguments"
                    state.messages.append(Turn("tooler", f"Attempt {attempt}: {last_err}"))
                    continue

                # Attach allowed runtime variables
                runtime_scope = {
                    "kern_text": kern_text,
                    "question_text": question_text,
                    "music_id": music_id,
                    "question_id": qid,
                }
                
                # Replace special tokens with actual values
                patched_args = {}
                for k, v in args.items():
                    if isinstance(v, str) and v.startswith("$") and v[1:] in runtime_scope:
                        patched_args[k] = runtime_scope[v[1:]]
                    else:
                        patched_args[k] = v

                # Execute tool with validation
                tool_schema = get_tool_schema(tools_yaml, fn)
                if not tool_schema:
                    last_err = f"Schema for tool '{fn}' not found"
                    state.messages.append(Turn("tooler", f"Attempt {attempt}: {last_err}"))
                    log_event(state, step, "tooler", "error", last_err, {"attempt": attempt})
                    # Send the error directly back to the Thinker and end this CALL_TOOL round
                    state.messages.append(Turn("thinker", f"TOOL_ERROR: {last_err}"))
                    log_event(state, step, "thinker", "error", last_err, {"attempt": attempt})
                    break

                # Auto-inject kern_data if missing/invalid (robustness against LLM omissions)
                try:
                    required_list = tool_schema.get('required', []) or []
                except Exception:
                    required_list = []

                if 'kern_data' in (tool_schema.get('properties', {}) or {}) or 'kern_data' in required_list:
                    kd = patched_args.get('kern_data')
                    need_inject = False
                    if not isinstance(kd, str):
                        need_inject = True
                    else:
                        # Require non-trivial content and presence of '**kern'
                        if len(kd.strip()) < 16 or ('**kern' not in kd):
                            need_inject = True
                    if need_inject:
                        patched_args['kern_data'] = kern_text
                        # Don't add auto-injection message to state.messages, only log it
                        log_event(state, step, "tooler", "info", "Auto-injected kern_data", {"attempt": attempt, "kern_len": len(kern_text)})

                # De-duplicate exact same tool call (same fn + same args), to avoid repeated work
                sanitized_args = {}
                for k2, v2 in patched_args.items():
                    if k2 == 'kern_data' and isinstance(v2, str):
                        sanitized_args[k2] = f"<len={len(v2)}>"
                    else:
                        sanitized_args[k2] = v2
                try:
                    call_key = f"{fn}:{json.dumps(sanitized_args, ensure_ascii=False, sort_keys=True)}"
                except Exception:
                    call_key = f"{fn}:{str(sorted(list(sanitized_args.keys())))}"

                # Check for recent duplicate calls (within last 3 calls)
                recent_duplicates = [call for call in state.recent_tool_calls[-3:] if call == call_key]
                if recent_duplicates:
                    # Only log to trace events, don't add to messages
                    log_event(state, step, "tooler", "info", f"Duplicate tool call detected for {fn} with same arguments (recently called {len(recent_duplicates)} times). Skipping execution.", {"attempt": attempt})
                    break

                # Log brief argument summary (without dumping full text) - only to trace events, not messages
                kd_len = len(patched_args.get('kern_data', '')) if isinstance(patched_args.get('kern_data'), str) else 0
                arg_keys = sorted(list(patched_args.keys()))
                log_event(state, step, "tooler", "info", f"Executing {fn} with args={arg_keys}, kern_data_len={kd_len}", {"attempt": attempt})
                log_event(state, step, "tool", "call", f"Executing {fn}", {"attempt": attempt, "args": sanitized_args, "kern_data_len": kd_len})

                result = execute_tool_with_validation(tools_mod, fn, patched_args, tool_schema)
                
                if result.success:
                    state.messages.append(Turn("tooler", f"Call specification: {json.dumps(call_spec, ensure_ascii=False)}"))
                    state.messages.append(Turn("tool", f"{fn} → {result.data}"))
                    log_event(state, step, "tool", "result", result.data, {"attempt": attempt, "function": fn, "success": True})
                    # Record successful call to prevent immediate duplicates
                    state.recent_tool_calls.append(call_key)
                    break
                else:
                    last_err = f"Tool execution failed: {result.error}"
                    # Change to feeding the error back to the Thinker
                    state.messages.append(Turn("thinker", f"TOOL_ERROR: {fn} failed (attempt {attempt}): {result.error}"))
                    log_event(state, step, "tool", "error", f"{fn} error: {result.error}", {"attempt": attempt, "function": fn})
                    log_event(state, step, "thinker", "error", f"{fn} error: {result.error}", {"attempt": attempt, "function": fn})
                    # Stop retrying and return control to the Thinker to re-plan in the next round
                    break
            else:
                # all attempts failed
                state.messages.append(Turn("thinker", f"TOOL_ERROR: Failed after {TOOL_RETRY} attempts. Last error: {last_err}"))
                log_event(state, step, "thinker", "error", f"All attempts failed: {last_err}")
                continue

        # Fallback (unknown decision) -- only when not a handled CALL_TOOL/THINK/FINISH
        if decision != "CALL_TOOL":
            state.messages.append(Turn("thinker", f"Unknown decision '{decision}'. Treating as THINK. {thought}"))

    # 3) Reviewer
    transcript = "\n".join(f"[{t.role}] {t.content}" for t in state.messages)
    reviewer_user_msg = REVIEWER_USER_TMPL.format(
        qid=qid,
        mid=music_id,
        question=question_text,
        transcript=transcript,
    )
    log_event(state, -1, "reviewer", "input/system", REVIEWER_SYS)
    log_event(state, -1, "reviewer", "input/user", reviewer_user_msg)
    reviewer_out_raw = ask_openai(
        OPENAI_MODEL_REVIEWER,
        REVIEWER_SYS,
        reviewer_user_msg,
    ).strip()
    log_event(state, -1, "reviewer", "output/model", reviewer_out_raw)

    # Extract answer from reviewer output
    pred = reviewer_out_raw.strip().upper()
    
    # Validate answer is A, B, C, or D
    if pred in {"A", "B", "C", "D"}:
        log_event(state, -1, "reviewer", "info", f"Answer: {pred}")
    else:
        pred = "A"  # safe default
        log_event(state, -1, "reviewer", "error", f"Invalid answer '{pred}', using default 'A'")

    state.messages.append(Turn("reviewer", pred))
    return pred, state

# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Run optimized MCQ music reasoning agent to process one question.")
    parser.add_argument("--question_id", "-q", type=str, default=None, help="Question ID in test.csv (e.g.: 001)")
    args = parser.parse_args()

    if args.question_id is None:
        # Use the first row if not specified
        with open(CSV_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
            if not first:
                raise RuntimeError("CSV file is empty")
            qid = str(first.get('question_id', '')).strip()
    else:
        qid = args.question_id

    pred, st = run_episode(qid, CSV_PATH)

    print("================ Results ================")
    print(f"Question ID: {st.question_id}")
    print(f"Music ID:    {st.music_id}")
    print(f"Prediction:   {pred}")
    print(f"Ground Truth: {st.answer_gt}")
    print(f"Correct?:    {pred == st.answer_gt}")
    print("==========================================")

    # Optionally dump transcript
    out_dir = os.environ.get("RUN_OUT_DIR", "runs")
    os.makedirs(out_dir, exist_ok=True)
    # Use question_id as music_id if music_id is empty (for reddit dataset)
    log_id = st.music_id if st.music_id else st.question_id
    with open(os.path.join(out_dir, f"{st.question_id}_{log_id}.log.txt"), "w", encoding="utf-8") as f:
        for t in st.messages:
            f.write(f"[{t.ts}][{t.role}] {t.content}\n\n")
        # Structured trace events for full step-by-step visibility
        f.write("================ TRACE EVENTS ================\n")
        for ev in st.trace_events:
            try:
                round_no = ev.get("round")
                phase = ev.get("phase")
                etype = ev.get("type")
                content = ev.get("content")
                meta = ev.get("meta")
                f.write(f"[round={round_no}][phase={phase}][type={etype}]\n")
                f.write(f"CONTENT:\n{content}\n")
                if meta is not None:
                    f.write(f"META:\n{json.dumps(meta, ensure_ascii=False, indent=2)}\n")
                f.write("--------------------------------------------\n")
            except Exception:
                # do not fail logging
                continue

    # Write structured JSON log with full reasoning trace
    json_log_obj = {
        "question_id": st.question_id,
        "music_id": st.music_id,
        "question": st.question,
        "ground_truth": st.answer_gt,
        "prediction": pred,
        "correct": pred == st.answer_gt,
        "messages": [
            {"role": t.role, "content": t.content, "ts": getattr(t, 'ts', None)}
            for t in st.messages
        ],
        "trace_events": st.trace_events,
    }
    with open(os.path.join(out_dir, f"{st.question_id}_{log_id}.log.json"), "w", encoding="utf-8") as jf:
        jf.write(json.dumps(json_log_obj, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
