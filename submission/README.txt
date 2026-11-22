GEMINI VLM RUN (Milestone 3 - VLM baseline, API version, model gemini-3.5-flash-lite)

You need: Python 3 (no extra packages), internet, and up to four free Gemini API keys.

1. Get keys: open https://aistudio.google.com , sign in, click "Get API key", create one key.
   Repeat for up to four keys (a different Google project gives a separate free quota).
   Keep keys private. Do not paste them into any file or chat.

2. Put this folder somewhere simple, e.g. Desktop\gemini_kit.

3. Open a terminal in this folder and set the keys (this terminal window only):
   Windows (Command Prompt):  set GEMINI_API_KEY=PASTE_KEY_ONE
                              set GEMINI_API_KEY_2=PASTE_KEY_TWO
                              set GEMINI_API_KEY_3=PASTE_KEY_THREE
                              set GEMINI_API_KEY_4=PASTE_KEY_FOUR
   Mac/Linux:                 export GEMINI_API_KEY=PASTE_KEY_ONE
                              export GEMINI_API_KEY_2=PASTE_KEY_TWO
                              export GEMINI_API_KEY_3=PASTE_KEY_THREE
                              export GEMINI_API_KEY_4=PASTE_KEY_FOUR

4. Run the clean charts, then the degraded charts (30 charts each, 6 per class):
   python run_vlm.py --manifest sample_manifest.json --out vlm_clean.jsonl
   python run_vlm.py --manifest sample_manifest.json --out vlm_degraded.jsonl --condition degraded

   If the free quota runs out, the script switches keys. If all keys are used up, it stops.
   Run the SAME command again later (or the next day): it resumes where it stopped.

5. Send the two files vlm_clean.jsonl and vlm_degraded.jsonl back in the chat.

If you see "GEMINI_API_KEY is not set", step 3 was not done in this terminal window.
If you see "Model not available (HTTP 404)", run with --model NAME, using a model name shown in AI Studio.

VISUAL STUDIO TERMINAL (Windows PowerShell)
Open the folder gemini_kit in Visual Studio (File > Open Folder), then open the terminal (View > Terminal).
Set the keys (PowerShell syntax; this terminal session only):
   $env:GEMINI_API_KEY="PASTE_KEY_ONE"
   $env:GEMINI_API_KEY_2="PASTE_KEY_TWO"
   $env:GEMINI_API_KEY_3="PASTE_KEY_THREE"
   $env:GEMINI_API_KEY_4="PASTE_KEY_FOUR"
Run:
   python run_vlm.py --manifest sample_manifest.json --out vlm_clean.jsonl
   python run_vlm.py --manifest sample_manifest.json --out vlm_degraded.jsonl --condition degraded
If "python" is not found, try "py" instead of "python".
Keep the terminal open between runs, or set the keys again after reopening it.
