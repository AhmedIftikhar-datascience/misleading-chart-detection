# Phase 3 implementation: AI-based detection of misleading charts

Code: code/. Results: results/ (JSON, CSV, OCR caches). Figures: figures/. Report: DAV_Phase3_Implementation_and_Evaluation.docx

Run order (from code/; paths assume the project layout under /home/claude/phase3):
1. python3 generate_charts.py ../data 300
2. CHART_SEED=99 CHART_ALL_TEST=1 python3 generate_charts.py ../data_confirm 200
3. python3 cache_ocr.py ../data ocr ; python3 cache_ocr.py ../data_confirm confirm
4. python3 run_final.py ../data ocr original_test ; python3 run_final.py ../data_confirm confirm confirm_test
5. python3 make_figures.py original_test orig ; python3 make_figures.py confirm_test conf
6. python3 build_report.py

Image files are regenerated from the seed and are not included in this archive.
The VLM review stage of the Phase 2 design is not implemented (no VLM was available in the execution environment).
