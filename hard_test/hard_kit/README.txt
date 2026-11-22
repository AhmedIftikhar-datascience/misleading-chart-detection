HARDER CHART SET - Gemini API run (model gemini-3.5-flash-lite)
50 charts per condition: 10 per class. Same prompt and keys setup as the earlier kit.

Keys: set the same environment variables as before (GEMINI_API_KEY, _2, _3, _4).
PowerShell example:
   $env:GEMINI_API_KEY="KEY_ONE"

Run these two commands (note --per-kind 10):
   python run_vlm.py --manifest sample_manifest.json --out vlm_hard_clean.jsonl --per-kind 10
   python run_vlm.py --manifest sample_manifest.json --out vlm_hard_degraded.jsonl --per-kind 10 --condition degraded

If quota runs out, run the same command again later; it resumes.
Send back vlm_hard_clean.jsonl and vlm_hard_degraded.jsonl.
