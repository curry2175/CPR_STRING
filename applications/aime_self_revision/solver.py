"""Small OpenAI Responses API wrapper used by the AIME self-revision loop."""
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from openai import OpenAI

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

def _load_env() -> None:
    for path in (REPO_ROOT / ".env", HERE / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            key,value=line.split("=",1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_env()

def _usage(response: Any) -> dict[str, Any]:
    usage=getattr(response,"usage",None)
    if usage is None: return {}
    if hasattr(usage,"model_dump"): return usage.model_dump()
    return {k:getattr(usage,k,None) for k in ("input_tokens","output_tokens","total_tokens")}

def _response_text(response: Any) -> str:
    text=str(getattr(response,"output_text","") or "").strip()
    if text: return text
    chunks=[]
    for item in getattr(response,"output",[]) or []:
        for content in getattr(item,"content",[]) or []:
            value=getattr(content,"text",None)
            if value: chunks.append(str(value))
    return "\n".join(chunks).strip()

def _call(prompt: str, *, model: str="gpt-5.4-nano", effort: str="low", max_output_tokens: int=3500, client: Any=None) -> dict[str, Any]:
    if client is None:
        from openai import OpenAI
        client=OpenAI()
    started=time.perf_counter()
    response=client.responses.create(model=model,input=prompt,reasoning={"effort":effort},max_output_tokens=max_output_tokens)
    return {"text":_response_text(response),"usage":_usage(response),"latency_s":round(time.perf_counter()-started,3),"response_id":getattr(response,"id",None)}
