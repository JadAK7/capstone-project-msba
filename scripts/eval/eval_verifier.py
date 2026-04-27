#!/usr/bin/env python3
"""
eval_verifier.py
Measure the effectiveness of the post-generation claim verifier.

For each golden-set question, the script:
  1. Runs the pipeline to get a grounded answer + context.
  2. Applies a programmatic corruption to the answer:
       - 'fabricate'       : append a sentence that the context cannot support.
       - 'number_swap'     : change the first numeric value to a different number.
       - 'entity_swap'     : replace a key proper-noun with a fake one.
  3. Calls verify_answer(query, corrupted, context).
  4. Records whether the verifier REMOVED the corruption (catch = 1) or kept it (miss = 0).

Also runs the verifier on the ORIGINAL answer to measure the false-positive rate
(does it mutilate a correct answer?).

Metrics:
  - catch_rate per corruption type = removed / (removed + kept)
  - false_positive_rate = fraction of ORIGINAL answers materially modified
  - behaviour on refusal answers (should be left alone)

Usage:
    python scripts/eval_verifier.py
    python scripts/eval_verifier.py --n 15
    python scripts/eval_verifier.py --types fabricate number_swap
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.database import init_db, close_db
from backend.chatbot import LibraryChatbot
from backend.verifier import verify_answer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DEFAULT_TYPES = ["fabricate", "number_swap", "entity_swap"]

FABRICATION_SENTENCE = (
    " The library also offers complimentary airline tickets to all enrolled "
    "students during the summer break."
)


def corrupt_answer(answer: str, corruption_type: str) -> tuple:
    """Return (corrupted_answer, injected_claim_text) or (None, None) if corruption cannot apply."""
    if corruption_type == "fabricate":
        corrupted = answer.rstrip() + FABRICATION_SENTENCE
        return corrupted, FABRICATION_SENTENCE.strip()

    if corruption_type == "number_swap":
        m = re.search(r"\b(\d+(?:\.\d+)?)\b", answer)
        if not m:
            return None, None
        original = m.group(1)
        swapped = str(int(float(original)) + 777) if "." not in original else "999.99"
        corrupted = answer[:m.start()] + swapped + answer[m.end():]
        return corrupted, f"numeric claim changed from '{original}' to '{swapped}'"

    if corruption_type == "entity_swap":
        # First try the curated list of library-specific entities — these
        # are the highest-signal swaps because the verifier should clearly
        # see the fake doesn't appear in context.
        swaps = [
            (r"\bJafet\b", "Farouk"),
            (r"\bScholarWorks\b", "ThesisLand"),
            (r"\bIEEE\b", "FAKE_IEEE"),
            (r"\bAUB\b", "ZZZ_Uni"),
            (r"\bSaab\b", "Foobar"),
            (r"\bAl Manhal\b", "FakeManhal"),
        ]
        for pat, repl in swaps:
            if re.search(pat, answer):
                corrupted = re.sub(pat, repl, answer, count=1)
                return corrupted, f"entity swapped: {pat} → {repl}"

        # Regex fallback: pick any capitalized 4+ letter token. Without
        # this, an answer that doesn't mention the curated nouns silently
        # skips the swap and the catch-rate is computed over a smaller,
        # biased subset (only questions where Jafet/IEEE/etc. happened to
        # appear). The fallback widens coverage to most answers.
        _STOPWORDS = {
            "The", "This", "That", "These", "Those", "There", "Their", "They",
            "Then", "Than", "When", "Where", "What", "Which", "While", "With",
            "Will", "From", "Have", "Here", "How", "However", "Also", "Note",
            "Please", "Library", "Libraries", "Students", "Student", "Faculty",
            "And", "But", "Because", "Since", "About", "After", "Before",
            "Within", "Without", "Between", "During", "Through", "Under",
            "Over", "More", "Most", "Some", "Many", "Much", "Other", "Another",
            "Each", "Every", "Both", "Either", "Neither",
        }
        # Find candidates sorted by position; prefer non-sentence-starters
        # so we don't pick "Library" at the start of every sentence.
        candidates = []
        for m in re.finditer(r"\b[A-Z][A-Za-z]{3,}\b", answer):
            tok = m.group(0)
            if tok in _STOPWORDS:
                continue
            # is_sentence_start: preceded by start-of-string OR ". " / ".\n"
            prefix = answer[max(0, m.start() - 2):m.start()]
            is_start = (m.start() == 0) or prefix.endswith(". ") or prefix.endswith(".\n")
            candidates.append((is_start, m.start(), tok))
        # Prefer non-starters, then earliest position
        candidates.sort(key=lambda c: (c[0], c[1]))
        if candidates:
            _, pos, tok = candidates[0]
            fake = "Fake" + tok
            corrupted = answer[:pos] + fake + answer[pos + len(tok):]
            return corrupted, f"entity swapped (regex fallback): {tok} → {fake}"
        return None, None
    raise ValueError(f"unknown corruption: {corruption_type}")


def detect_removal(verified: str, corrupted: str, injected: str, ctype: str) -> bool:
    """Heuristic: did the verifier remove / blunt the injection?"""
    if ctype == "fabricate":
        # The fabricated sentence key phrase: "airline tickets"
        return "airline tickets" not in verified.lower()
    if ctype == "number_swap":
        # injected contains e.g. "changed from '2' to '779'"
        m = re.search(r"to '([^']+)'", injected)
        if not m:
            return False
        swapped_value = m.group(1)
        # If the swapped number is no longer present OR the answer looks like a
        # refusal ("could not find" / "could not be verified"), count as removed.
        if swapped_value not in verified:
            return True
        lowered = verified.lower()
        if "could not find" in lowered or "could not be verified" in lowered:
            return True
        return False
    if ctype == "entity_swap":
        # injected contains e.g. "FAKE_IEEE" after the arrow
        m = re.search(r"→ (.+)$", injected)
        if not m:
            return False
        fake = m.group(1).strip()
        return fake not in verified


def is_material_mutation(original: str, verified: str) -> bool:
    """Did the verifier meaningfully change a correct answer? Token-level length delta >20%."""
    orig = original.strip()
    ver  = verified.strip()
    if not orig:
        return False
    if ver == orig:
        return False
    # Common disclaimer append is not a material mutation
    if ver.startswith(orig):
        appendix = ver[len(orig):].strip()
        if appendix.lower().startswith(("*note", "\n\n*note")):
            return False
    # Length-based heuristic
    if abs(len(ver) - len(orig)) / max(len(orig), 1) > 0.20:
        return True
    return False


def _is_refusal(answer: str, source: str) -> bool:
    return (source.startswith("none")
            or "could not find" in answer.lower()
            or "لم أتمكن" in answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str,
                        default=os.path.join(DATA_DIR, "golden_set.json"))
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--types", nargs="+", default=DEFAULT_TYPES, choices=DEFAULT_TYPES)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set."); sys.exit(1)

    with open(args.golden, encoding="utf-8") as f:
        data = json.load(f)
    candidates = [q for q in data["questions"]
                  if q.get("expected_behavior") == "answer" and q.get("language") == "en"]
    import random; random.seed(42)
    questions = random.sample(candidates, min(args.n, len(candidates)))

    init_db()
    chatbot = LibraryChatbot(os.environ["OPENAI_API_KEY"])

    per_q = []
    print(f"\n  Running pipeline + verifier on {len(questions)} questions...\n")
    for i, q in enumerate(questions, 1):
        t0 = time.time()
        try:
            answer, debug = chatbot.answer(q["question"])
        except Exception as e:
            print(f"    [err] pipeline failed on {q['id']}: {e}")
            continue

        source = debug.get("chosen_source", "")
        context = debug.get("context_sent_to_llm", "")
        if _is_refusal(answer, source) or not context:
            per_q.append({"id": q["id"], "question": q["question"],
                          "original_answer": answer, "source": source,
                          "skipped_reason": "refusal_or_no_context"})
            continue

        record = {
            "id": q["id"],
            "question": q["question"],
            "original_answer": answer,
            "source": source,
            "context_len": len(context),
            "corruptions": {},
        }

        # False-positive test: run verifier on ORIGINAL answer
        try:
            verified_orig, removed_orig = verify_answer(q["question"], answer, context, lang="en")
        except Exception as e:
            verified_orig, removed_orig = answer, [f"ERROR: {e}"]
        record["verified_original"] = verified_orig
        record["fp_mutation"]       = is_material_mutation(answer, verified_orig)
        record["removed_on_original"] = removed_orig

        # Corruption tests
        for ctype in args.types:
            corrupted, injected = corrupt_answer(answer, ctype)
            if corrupted is None:
                record["corruptions"][ctype] = {"skipped": True}
                continue
            try:
                verified, removed = verify_answer(q["question"], corrupted, context, lang="en")
                caught = detect_removal(verified, corrupted, injected, ctype)
            except Exception as e:
                verified, removed, caught = corrupted, [f"ERROR: {e}"], False
            record["corruptions"][ctype] = {
                "injected": injected,
                "caught": caught,
                "verified": verified,
                "removed_claims": removed,
            }

        per_q.append(record)
        elapsed = time.time() - t0
        print(f"    [{i:>2}/{len(questions)}] {q['id']:<20} fp_mutation={record['fp_mutation']} "
              + "  ".join(f"{c}={'Y' if record['corruptions'].get(c, {}).get('caught') else 'N'}"
                          for c in args.types) + f"  ({elapsed:.1f}s)")

    close_db()

    # Aggregate
    scored = [r for r in per_q if "corruptions" in r]
    n = len(scored)
    summary = {"n_scored": n, "catch_rates": {}}
    for ctype in args.types:
        applicable = [r for r in scored if not r["corruptions"].get(ctype, {}).get("skipped")]
        caught = sum(1 for r in applicable if r["corruptions"].get(ctype, {}).get("caught"))
        total = len(applicable)
        # Surface coverage so the catch_rate denominator shrinkage is
        # explicit. If a corruption only applies to half the answers, the
        # catch_rate is computed over that half — readers should know.
        summary["catch_rates"][ctype] = {
            "applicable": total,
            "n_total_scored": n,
            "coverage": round(total / n, 4) if n else 0.0,
            "caught": caught,
            "catch_rate": round(caught / total, 4) if total else None,
        }
    fp_count = sum(1 for r in scored if r["fp_mutation"])
    summary["false_positive_rate"] = round(fp_count / n, 4) if n else 0.0
    summary["false_positive_n"] = fp_count

    # Report
    print("\n" + "=" * 72)
    print("  VERIFIER EFFECTIVENESS")
    print("=" * 72)
    print(f"  n scored = {n}\n")
    print(f"  Catch rate by corruption type:")
    for ctype, m in summary["catch_rates"].items():
        if m["catch_rate"] is None:
            print(f"    {ctype:<14} (no applicable questions of {m['n_total_scored']})")
        else:
            cov_warn = "  ⚠ low coverage" if m["coverage"] < 0.5 else ""
            print(
                f"    {ctype:<14} {m['caught']:>2}/{m['applicable']:<2}  "
                f"({m['catch_rate']:.0%})   "
                f"coverage={m['coverage']:.0%} of {m['n_total_scored']}{cov_warn}"
            )
    print(f"\n  False-positive rate (original answer materially mutated): "
          f"{summary['false_positive_rate']:.1%}  ({fp_count}/{n})")
    print("  → lower is better (verifier should leave correct answers alone)")

    out = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_questions": n,
            "corruption_types": args.types,
        },
        "summary": summary,
        "per_question": per_q,
    }
    out_path = args.output
    if not out_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            f"eval_verifier_{ts}.json",
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
