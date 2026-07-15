"""Offline LLM judge for the M9b detector — recovers dictionary-word names.

The code-only detector drops clusters where every spelling is an ordinary word,
which loses improvised names that are also dictionary words (Bibi, Bacchus). This
hands exactly those ambiguous clusters to Qwen 3.5 4B and asks, of the flagged
spellings themselves, whether at least one is a proper name (invented OR borrowed)
rather than ordinary words.

Qwen replaced the earlier Gemma-4 E4B judge once a Qwen-fit prompt was found: it
matches Gemma on the M9b answer key (1.00/1.00) and on a held-out synthetic set,
so the whole pipeline now runs one local model. The tuning story (Gemma wanted
worked examples; Qwen wanted a broad knowledge-based question with the over-narrow
words removed) lives in emp/src/judge_m9b.py and emp/writeup/name-consistency-eval.html.

`make_judge()` returns judge(candidates) -> set[cluster_id]. The model runs via the
shared model_runner — a fresh subprocess of this venv, a clean process so a pyannote
MPS allocation can't block the model load, freeing GPU memory on exit.

It is NEVER called from the live API (a ~30s model load can't sit in a page-view
request). Only `detect.py --judge` supplies it; the result survives normal Monitor
viewing and is reverted to code-only if the transcript changes (re-run to restore).
"""
from collections import Counter

from model_runner import run_model

RUNS = 3

# The M9b judge prompt, tuned for Qwen 3.5 4B. Qwen follows instructions literally,
# so it wants a broad, knowledge-based question — judge the spelling itself, and let
# a *borrowed* name (Bacchus, a real myth name a child reused) count — rather than
# Gemma's worked-example scaffolding (every Gemma-style addition lowered Qwen here).
# This exact wording scores 1.00/1.00 on the M9b key and matches Gemma on a held-out
# synthetic set; tuned in emp/src/judge_m9b.py. Keep it verbatim or re-validate.
_NAME_DEF = (
    "a proper NAME of a character, creature, toy, or figure in the story — whether "
    "the child invented it or borrowed it from a book, show, or myth"
)
PROMPT = (
    "You are checking a transcript of a parent telling a young child a bedtime "
    "story. A group of similar-sounding capitalized words was flagged because the "
    "transcriber may have spelled one name several different ways.\n\n"
    "Look at the flagged spellings themselves. Decide whether at least ONE of them "
    "is " + _NAME_DEF + ", as opposed to ordinary English words, interjections (Oh, "
    "Yeah, Whoa), question words (What, There), or family terms (Daddy, Mommy). A "
    "name still counts when a similar-sounding ordinary word is mixed into the "
    "group; the example lines are only context.\n\n"
    "Flagged spellings: {spellings}\n"
    "Example lines:\n{sentences}\n\n"
    "Reply with ONLY one word and nothing else: NAME if at least one spelling is "
    "such a name, or WORD if all of them are ordinary."
)


def _judge_clusters(candidates):
    """Module-level (picklable for the spawned subprocess). Loads Qwen 3.5 4B once
    and returns the cluster_ids that majority-vote to NAME. Each candidate:
    {cluster_id, spellings: [...], examples: [...]}."""
    from qwen35 import make_reader

    gen = make_reader()
    kept = []
    for c in candidates:
        prompt = PROMPT.format(spellings=", ".join(c["spellings"]),
                               sentences="\n".join(f"- {s}" for s in c["examples"]))
        verdicts = []
        for _ in range(RUNS):
            up = gen(prompt, max_tokens=8).upper()
            verdicts.append("NAME" if ("NAME" in up and "WORD" not in up) else "WORD")
        if Counter(verdicts).most_common(1)[0][0] == "NAME":
            kept.append(c["cluster_id"])
    return kept


def make_judge():
    """Return judge(candidates) -> set[cluster_id], or raise FileNotFoundError if mlx-vlm
    isn't installed (callers degrade to code-only on that). Runs Qwen in a fresh
    subprocess (via model_runner) so the ~30s model load gets a clean GPU process and
    frees it on exit. Each candidate: {cluster_id, spellings: [...], examples: [...]}."""
    import importlib.util
    if importlib.util.find_spec("mlx_vlm") is None:
        raise FileNotFoundError(
            "mlx-vlm is not installed in this venv; the M9b judge needs it "
            "(pip install 'mlx-vlm==0.5.0'). Caller falls back to code-only.")

    def judge(candidates):
        return set(run_model(_judge_clusters, candidates, timeout=600))

    return judge
