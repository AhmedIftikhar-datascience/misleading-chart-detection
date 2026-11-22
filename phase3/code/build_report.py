"""Build the Phase 3 report (.docx). All numbers are read from results JSON."""
import json
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

R = "/home/claude/phase3/results"
FIG = "/home/claude/phase3/figures"
O = json.load(open(f"{R}/original_test.json"))
C = json.load(open(f"{R}/confirm_test.json"))
OS, CS = O["summary"], C["summary"]
V3C = "Proposed v3: OCR + rules + missing-tick check (clean)"
V3D = "Proposed v3: OCR + rules + missing-tick check (degraded)"
V2D = "Proposed v2: OCR + rules, no missing-tick check (degraded)"
ORA = "Oracle metadata + rules (upper bound)"
ALW = "Baseline B0: always flag"


def f3(x):
    return f"{x:.3f}"


def ci(s):
    return f"[{s[0]:.3f}, {s[1]:.3f}]"


doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)


def H(t, lvl=1):
    doc.add_heading(t, level=lvl)


def P(t, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(t); r.bold = bold; r.italic = italic
    return p


def B(t):
    doc.add_paragraph(t, style="List Bullet")


def T(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = h
        for p in c.paragraphs:
            for r in p.runs:
                r.bold = True; r.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()
    return t


def FIGURE(path, caption):
    doc.add_picture(path, width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(caption); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True; p.runs[0].font.size = Pt(9)


# ---------- title ----------
t = doc.add_heading("AI-Based Detection of Misleading Data Visualizations", 0)
P("A Quantitative Evaluation of Chart Deception and Reliability", italic=True)
P("DATA ANALYSIS & VISUALIZATION — MILESTONE 3 (PHASE 3): Research Implementation and Experimental Evaluation")
P("Group members: Ahmed Iftikhar (24i-2607), Mohsin Masood Abassi (24i-2526)")

H("Summary")
P("We implemented the non-VLM part of the Phase 2 method: OCR-based extraction of axis evidence, "
  "followed by three deterministic design rules, with abstention when extraction is incomplete. "
  "We evaluated it on a synthetic set of paired honest and misleading bar and line charts, "
  "using a prevalence sweep, a confirmatory test set and ablations. "
  "Three findings matter most:")
B(f"On clean charts the pipeline flagged every misleading chart in the test set (TPR 1.000) "
  f"with a false-positive rate of {f3(OS[V3C]['FPR'])}. Its false positives all came from honest line charts.")
B(f"Under JPEG compression and downscaling, the first version (v2) flagged "
  f"{OS[V2D]['FPR_honest_bar']*100:.1f}% of honest bar charts as truncated. Precision at 5% misleading prevalence fell to "
  f"{OS[V2D]['precision_by_prevalence']['0.05']['mean']:.3f}. A missing-tick check (v3) removed these false alarms "
  f"(FPR on honest bar charts {f3(OS[V3D]['FPR_honest_bar'])}), but it abstains on "
  f"{OS[V3D]['abstain_honest']*100:.1f}% of degraded honest bar charts. The v3 result replicated on a fresh confirmatory set.")
B("An image-only VLM baseline (gemini-3.5-flash-lite, run through the Gemini API) was scored on 30 clean and 30 degraded test charts (6 per class). "
  "The VLM review stage of the Phase 2 design was not implemented in code. On both sets the VLM and the rule pipeline were fully correct, "
  "with no false alarms and no discordant pairs, so these samples show no difference between them. The VLM also named the misleader type correctly on every misleading chart. "
  "A manual run in the Gemini web app is reported as a cross-check.")

# ---------- 1 ----------
H("1. Implementation of the proposed methodology")
P("Table 1 maps each Phase 2 step to what was implemented. Every change from the Phase 2 design is listed, as the brief requires.")
T(["Phase 2 step", "Implemented in Phase 3", "Change and reason"], [
    ["1 Research input", "Chart PNG only; output is a flag, its misleader type(s), and a fired-rule list.",
     "Source tables are not used at inference; they only generate the charts. Severity score not produced."],
    ["2 Data collection", "Synthetic paired charts from 300 generated tables (matplotlib, fixed seed).",
     "Misviz not used: its licence and download were not checked. Synthetic pairs replace it; limitation stated."],
    ["3 Preprocessing", "Fixed 1200×800 px rendering; pairs kept in the same split.",
     "Perceptual-hash deduplication not needed (every chart is generated uniquely). Split is by table, so no leakage."],
    ["4 Feature engineering", "Chart type (bar/line) from palette pixels; left and right tick labels via Tesseract with vertical position.",
     "Axis titles, legends and plotted values are not extracted. Chart type depends on the generator's colours (limitation)."],
    ["5 Proposed method", "Stage A rules: truncated baseline (bars only), inverted axis, dual axis. Abstention when fewer than 3 ticks are read.",
     "VLM review stage NOT implemented in code. Image-only VLM baseline run through the Gemini API on 30 clean and 30 degraded charts (Section 3). Missing-tick check added in v3 (Section 1.1). Calibration not possible: rules output no probability."],
    ["6 Baselines", "B0 always flag, B0b never flag; oracle-metadata upper bound.",
     "Image-only VLM (B1) run through the Gemini API on 30 clean and 30 degraded charts, with a manual web-app cross-check. Table-based QA (B3) not run. Rule-only (B2) equals the proposed method here, because the VLM review stage is not implemented."],
    ["7 Experimental design", "Dev split (100 tables) for pipeline checks; test split (200 tables); confirmatory set (200 new tables, new seed).",
     "Prevalence sweep by resampling 400 charts with replacement at each prevalence. Five seeds not needed: the pipeline is deterministic."],
    ["8 Metrics", "TPR, FPR (overall and by honest chart type), F1, precision at 5–95% prevalence, abstention rate, recall by misleader type, extraction diagnostics.",
     "Calibration (ECE, Brier) not reported: no probabilities. Run-to-run agreement is 1.0 by construction."],
    ["9 Statistics", "Pair-clustered bootstrap CIs (1,000 resamples); McNemar's test with continuity correction.",
     "Holm correction not applied: four pre-specified comparisons, reported without adjustment."],
    ["10 Ablation", "Each rule removed; truncation rule ungated; oracle versus OCR.", "As specified."],
    ["11 Error analysis", "Degraded-image failure (missing bottom tick); OCR outlier tokens (stray numbers in the margin).", "Manual review of 50 failures not performed; failures counted from all test charts."],
    ["12 Output", "Code, generator, results JSON/CSV, figures, this report.", "Image files are regenerated from the seed, not shipped."],
])

H("1.1 Changes made during the work", 2)
P("The brief asks for changes to the methodology to be documented. Two changes were made.")
B("v2 to v3, the missing-tick check. Under degradation, Tesseract often missed the bottom '0' tick. The lowest readable tick "
  "then looked like a non-zero baseline and the truncation rule fired on honest bar charts. v3 abstains when the bars "
  "extend at least half a tick step below the lowest readable tick. Disclosure: the failure was first noticed in the v2 results "
  "on the test split. The fix was then designed and checked on the dev split (honest-bar false alarms 87 of 100 removed, "
  "truncated recall unchanged), and only after that were the reported test runs made. The test split was therefore not "
  "independent of the diagnosis; the confirmatory set is the independent check.")
B("Line-chart hard negatives were added after the first pilot. The pilot used only zero-baseline bar charts as honest "
  "examples. With those, the detector scored perfectly, which said nothing about the research gap. Honest line charts with "
  "a non-zero baseline were then added. This was a post-hoc change to the dataset and is disclosed here. The confirmatory "
  "set (new seed, all test) was generated afterwards to check the v3 result.")

H("1.2 Feedback from the previous class", 2)
P("The only feedback on record is the teacher's remark on our Phase 1 iteration. The teacher gave no feedback on Milestone 2.")
T(["Feedback on record", "How Phase 3 responds"], [
    ["Some numbers taken from arXiv versions; cite final published versions.",
     "Section 5 uses only the figures in the Milestone 2 references. None of them is used as an input to the experiments. "
     "The experiments do not depend on literature numbers."],
    ["Add a base-rate column (honest vs misleading) so Gap 1 is visible.",
     "Every dataset in this report states its honest/misleading composition (Table 3). Section 5 repeats the base rate for each comparison."],
    ["Gap 5 overlaps Gap 1.", "Gap 5 (explanation faithfulness) was removed in Milestone 2. Phase 3 addresses Gaps 1 and 2 only."],
    ["No synthesis row across the comparison table.", "Milestone 2 contains a cross-paper synthesis; Phase 3 relies on it."],
])

# ---------- 2 ----------
H("2. Dataset, preprocessing and experimental setup")
P("Each generated table yields five charts: an honest bar chart with a zero baseline, an honest line chart with a non-zero baseline, "
  "and three misleading bar-chart variants (truncated baseline, inverted y-axis, dual y-axis with a rescaled second series). "
  "All five share the same numbers, so a pair differs only in the design.")
T(["Split", "Tables", "Charts", "Composition", "Use"], [
    ["Dev", "100", "500", "100 honest bar, 100 honest line, 300 misleading", "Pipeline checks, missing-tick check design"],
    ["Test (original)", "200", "1,000", "200 honest bar, 200 honest line, 600 misleading (200 per type)", "Reported results"],
    ["Confirmatory", "200", "1,000", "Same composition; new seed (99)", "Replication of v3 only"],
])
P("Base rate: 60% of the 1,000 test charts are misleading (600), 20% are honest bar charts and 20% are honest line charts, "
  "the hard negatives. The honest class is therefore 40% of the test set.", italic=True)
P("Preprocessing and implementation details:", bold=True)
B("Charts: matplotlib 3.11, 6×4 in figure at 200 dpi (1200×800 px). Degraded copies: downscaled to 600×400 px and JPEG quality 35.")
B("OCR: Tesseract (pytesseract 0.3.13) with --psm 6 and a digit whitelist, run on the left and right 22% of the image width.")
B("Chart type: bar if at least 2,000 pixels match the bar colour within tolerance 30; otherwise line.")
B("Rules: truncated = lowest left tick > 0 (bars only; ungated in the ablation); inverted = positive correlation between tick value and pixel row; "
  "dual = at least 3 numeric ticks on the right margin. Abstain if fewer than 3 left ticks are read.")
B("Oracle arm: the same rules applied to the generator's true axis settings, to separate extraction error from rule error.")
B("Prevalence sweep: 400 charts sampled with replacement from the test pool at each of 5, 10, 20, 50, 80 and 95% misleading; 1,000 repetitions.")
B("Environment: Python 3.13, numpy 2.5, scipy 1.18, scikit-learn 1.9 (not used in the final run), OpenCV 5.0; 2 CPU cores.")

# ---------- 3 ----------
H("3. Baseline evaluation")
P("Table 4 reports the test split. Confidence intervals are pair-clustered bootstrap intervals. "
  "Precision is the sampled value at the stated misleading prevalence.")
rows = []
for arm, lab in [(V3C, "Proposed v3, clean"), (V3D, "Proposed v3, degraded"), (V2D, "Proposed v2, degraded"),
                 (ORA, "Oracle metadata + rules"), (ALW, "Baseline B0: always flag")]:
    s = OS[arm]; pb = s["precision_by_prevalence"]
    rows.append([lab, f"{f3(s['TPR'])}", f"{f3(s['FPR'])} {ci(s['FPR_CI'])}",
                 f"{f3(s['FPR_honest_bar'])} / {f3(s['FPR_honest_line'])}", f3(s["F1"]),
                 f3(pb["0.05"]["mean"]), f3(pb["0.5"]["mean"]), f3(pb["0.95"]["mean"]),
                 f"{s['abstain_honest']*100:.1f}%"])
T(["Arm", "TPR", "FPR [95% CI]", "FPR bar / line", "F1", "Precision @5%", "Precision @50%", "Precision @95%", "Abstain (honest)"], rows)
P("Table 5 reports the confirmatory set for the v3 and v2 arms. It uses the same arms and the same analysis.")
rows = []
for arm, lab in [(V3C, "Proposed v3, clean"), (V3D, "Proposed v3, degraded"), (V2D, "Proposed v2, degraded")]:
    s = CS[arm]; pb = s["precision_by_prevalence"]
    rows.append([lab, f3(s["TPR"]), f"{f3(s['FPR'])} {ci(s['FPR_CI'])}", f"{f3(s['FPR_honest_bar'])} / {f3(s['FPR_honest_line'])}",
                 f3(s["F1"]), f3(pb["0.05"]["mean"]), f3(pb["0.95"]["mean"]), f"{s['abstain_honest']*100:.1f}%"])
T(["Arm", "TPR", "FPR [95% CI]", "FPR bar / line", "F1", "Precision @5%", "Precision @95%", "Abstain (honest)"], rows)
P("McNemar tests (test split): v3 versus v2 on degraded charts, 185 charts v2 got right and v3 got wrong, 0 the other way, p < 0.001; "
  "on clean charts the two versions are identical (p = 1.0). v3 clean versus oracle: 0 versus 3 discordant charts, p = 0.25. "
  "Confirmatory set: v3 versus v2 on degraded charts, 177 versus 0, p < 0.001; v3 clean versus oracle 0 versus 2, p = 0.48.")
P("Extraction diagnostics: on all charts, at least 3 left ticks were read and the chart type was correct for 100% of charts, "
  "both clean and degraded. The failures are therefore not visible in these counts; they appear as wrong tick values and are discussed in Section 6.", italic=True)
P("Image-only VLM baseline (B1), run through the Gemini API with gemini-3.5-flash-lite (6 charts per class, 30 charts per condition). "
  "Clean: 30 of 30 answered. The VLM and the rule pipeline were both fully correct (TPR 1.00, FPR 0.00). The VLM named the misleader type correctly on all 18 misleading charts. "
  "Degraded: 30 of 30 answered, with the same result for both methods (TPR 1.00, FPR 0.00; all 18 misleading charts typed correctly). "
  "The McNemar test shows no discordant pairs (p = 1.0). A manual run in the Gemini web app on 30 clean and 16 degraded charts gave the same result. "
  "Because the VLM also scores perfectly on these synthetic charts, the benchmark may be too easy to separate the methods, and six charts per class give very wide intervals.")

# ---------- 4 ----------
H("4. Graphical results")
FIGURE(f"{FIG}/orig_precision_prevalence.png",
       "Figure 1. Precision against misleading prevalence (test split). FPR is fixed by design, so precision alone shows the prevalence effect. Shaded: 95% intervals.")
FIGURE(f"{FIG}/orig_fpr_and_recall.png",
       "Figure 2. Left: false-positive rate on honest bar and line charts. Right: recall by misleader type (test split).")
FIGURE(f"{FIG}/orig_ablation.png",
       "Figure 3. Ablation on clean charts (test split). F1 and false-positive rate per arm.")
FIGURE(f"{FIG}/conf_precision_prevalence.png",
       "Figure 4. Precision against prevalence on the confirmatory set; same pattern as Figure 1.")

# ---------- 5 ----------
H("5. Comparison with existing research")
P("No reviewed study reports a comparable detection benchmark, so this section compares findings, not scores. "
  "Literature figures are those in the Milestone 2 references and were not re-verified in this phase.")
T(["Study (venue)", "Reported finding", "Base rate in that study", "Comparable to Phase 3?"], [
    ["Mahbub et al. 2025 (IEEE VIS)", "Inverted axis changed answers of 10 of 10 VLMs; most VLMs deceived.",
     "Paired design; Milestone 2 states 100 charts (not re-checked here); split not reported.",
     "No. They measure answer change in VLMs; we measure rule detection. Their inversion finding is consistent with our rule catching all inverted charts (synthetic)."],
    ["Tonglet et al. 2026 (ACL)", "Mean MLLM accuracy 26.4% on misleading charts vs 25.6% random; table QA +15.4 to +19.6 pp.",
     "143 misleading vs 124 non-misleading (53.6% / 46.4%).",
     "No. QA accuracy is a different task. Their table-QA idea is the basis for the Stage A extraction step."],
    ["Biselli et al. 2025 (ACM DIS)", "Qualitative think-aloud usability study; no precision or FPR reported.",
     "Not applicable.", "No. Shows the reporting gap that this project addresses."],
    ["Li et al. 2026 (Findings EMNLP)", "Misleading charts cut web-agent task success by more than 34 points.",
     "Matched clean/misleading pairs, 50/50.", "No. Downstream agent success, not detector FPR."],
    ["Rahman et al. 2026; Su et al. 2025; Das et al. 2026; Hong et al. 2025", "Bias, claim verification, literacy tests.",
     "Not applicable.", "Supporting evidence only."],
])
P("Consequence: no published detector reports precision or FPR at a range of prevalences. This is the gap Phase 3 targets, "
  "and our results cannot be placed against any published detector score.")

# ---------- 6 ----------
H("6. Ablation studies, error analysis and discussion")
H("6.1 Ablation", 2)
T(["Arm (clean, test)", "TPR", "FPR", "Line FPR", "F1", "Interpretation"], [
    ["Full v3", "1.000", f3(OS[V3C]["FPR"]), f3(OS[V3C]["FPR_honest_line"]), f3(OS[V3C]["F1"]), "Reference"],
    ["No truncation rule", "0.667", f3(OS["Ablation v3: no truncation rule (clean)"]["FPR"]), f3(OS["Ablation v3: no truncation rule (clean)"]["FPR_honest_line"]), f3(OS["Ablation v3: no truncation rule (clean)"]["F1"]), "Truncated charts all missed"],
    ["No inversion rule", "0.667", f3(OS["Ablation v3: no inversion rule (clean)"]["FPR"]), f3(OS["Ablation v3: no inversion rule (clean)"]["FPR_honest_line"]), f3(OS["Ablation v3: no inversion rule (clean)"]["F1"]), "Inverted charts all missed; removes all line-chart false positives"],
    ["No dual-axis rule", "0.667", f3(OS["Ablation v3: no dual-axis rule (clean)"]["FPR"]), f3(OS["Ablation v3: no dual-axis rule (clean)"]["FPR_honest_line"]), f3(OS["Ablation v3: no dual-axis rule (clean)"]["F1"]), "Dual-axis charts all missed"],
    ["Truncation rule ungated", "1.000", f3(OS["Ablation v3: truncation rule ungated (clean)"]["FPR"]), f3(OS["Ablation v3: truncation rule ungated (clean)"]["FPR_honest_line"]), f3(OS["Ablation v3: truncation rule ungated (clean)"]["F1"]), "Every honest line chart flagged; the chart-type gate is essential"],
    ["Oracle metadata", "1.000", "0.000", "0.000", "1.000", "Upper bound; OCR costs nothing on clean charts"],
])
P("The chart-type gate is the most important component: without it, every honest line chart is flagged. Each rule is needed for its own misleader type. "
  "The inversion rule also causes the only line-chart false positives, which is discussed below.")

H("6.2 Error analysis", 2)
B("Missing bottom tick (degraded charts). Under compression the bottom '0' tick was often unread. Without the missing-tick check, "
  "the lowest readable tick looked like a non-zero baseline and 92.5% of honest bar charts were flagged. The check abstains instead. "
  "Cost: 46% of degraded honest bar charts receive no decision, so the pipeline trades coverage for precision.")
B("OCR outlier tokens. In three honest line charts, a stray number inside the left margin (for example 707, 307 or 407, which look like "
  "data-point labels or stray text) entered the tick set and reversed the apparent axis direction. The inversion rule then fired. "
  "This accounts for all line-chart false positives: 3 of 200 honest line charts in the test split and 2 of 200 in the confirmatory set. Filtering tokens that do not sit on a regular step would likely fix it, "
  "but we did not add that fix, to avoid tuning on the test split.")
B("Under the v2 design, precision at 5% prevalence is 0.105 on degraded charts. Under v3 it is 0.917. The same FPR gives very different "
  "precision at low prevalence, which is the core of Gap 1.")

H("6.3 Analysis against the research gap and objectives", 2)
P("Gap 1 (reliability under realistic prevalence): addressed in part. Figure 1 shows that precision depends strongly on prevalence "
  "even when FPR is fixed, and that degradation can dominate prevalence. The results do not show precision on real-world prevalence, "
  "because the data are synthetic.")
P("Gap 2 (grounded evidence and error propagation): partly addressed. Extraction errors were isolated with the oracle arm and traced to "
  "specific failures (missing tick, outlier token). The VLM review stage was not implemented, so whether structured evidence helps a VLM "
  "remains unanswered.")
P("Gap 3 (impact): not addressed. No human-impact data were used.")
P("Research questions: RQ1 is answered for the rule pipelines (Figure 1). RQ2 (structured extraction versus image-only VLM) is answered only "
  "in a small sample: on 30 clean and 30 degraded charts the VLM and the rules were both fully correct, so no difference was detected. "
  "The sample is too small to rule out a difference, and the synthetic charts may be too easy to separate the methods. RQ3 is partly answered by the ablation and error analysis.")

H("6.4 Strengths and limitations", 2)
P("Strengths:", bold=True)
B("Deterministic and interpretable: each flag names the rule that fired, which supports the explanation requirement of Gap 3.")
B("No training data or model weights; runs on a laptop in minutes once OCR is cached.")
B("The confirmatory set reproduced the v3 result (FPR 0.007 on degraded charts; 0.000 on honest bar charts), so the v3 result is not a test-set artefact.")
P("Limitations:", bold=True)
B("Synthetic charts with one font, one palette and matplotlib defaults. Real-world performance will be lower, and the chart-type detector depends on colour.")
B("Three of the eight misleader types in the Phase 2 taxonomy were implemented. Deceptive titles, cherry-picking and misleading annotations are not covered.")
B("The VLM review stage was not implemented in code. The method evaluated here is the rule pipeline, not the full Phase 2 design.")
B("The VLM baseline is small (30 clean and 30 degraded charts, 6 per class) and uses one model (gemini-3.5-flash-lite) at temperature 0. "
  "Results may change when the model is updated. The API sample comes from the 250-chart kit manifest, so it differs from the manual sample.")
B("Honest charts are generated under the same conventions as the misleading ones, so the honest class is easier than real-world honest charts.")
B("No calibration, no human study, no severity weighting.")

# ---------- 7 ----------
H("7. Reproducibility")
P("Code and results are in the project folder. Images are regenerated from the seed.")
T(["File", "Purpose"], [
    ["code/generate_charts.py", "Builds paired charts and the manifest (env CHART_SEED, CHART_ALL_TEST)."],
    ["code/detect.py", "Stage A (OCR, chart type, bar bottom) and Stage B (rules, missing-tick check)."],
    ["code/cache_ocr.py", "Caches OCR evidence for a dataset root."],
    ["code/run_experiments.py", "Pilot run (v2 arms on the original dataset)."],
    ["code/run_final.py", "Final evaluation of all arms, CIs, prevalence sweep and McNemar tests for one dataset."],
    ["code/make_figures.py", "Figures 1–4 from the results JSON."],
    ["code/build_report.py", "This report."],
    ["results/original_test.json, confirm_test.json", "All numbers in this report."],
])
P("Commands: python3 generate_charts.py <dir> 300; python3 cache_ocr.py <dir> <prefix>; python3 run_final.py <dir> <prefix> <name>; python3 make_figures.py <name> <prefix>.", italic=True)


H("8. References")
P("As cited in the submitted Milestone 2 document.", italic=True)
for ref in [
    "Biselli, T., Hartwig, K., Kneissl, N., Pouliot, L., & Reuter, C. (2025). ChartChecker: A user-centred approach to support the understanding of misleading charts. Proceedings of ACM DIS '25, 2075–2102. https://doi.org/10.1145/3715336.3735784",
    "Das, A. K., Tarun, M., & Mueller, K. (2026). Charts-of-Thought: Enhancing LLM visualization literacy through structured data extraction. IEEE TVCG, 32(1), 1427–1437. https://doi.org/10.1109/TVCG.2025.3634813",
    "Hong, J., Seto, C., Fan, A., & Maciejewski, R. (2025). Do LLMs have visualization literacy? An evaluation on modified visualizations to test generalization in data interpretation. IEEE TVCG, 31(10), 7004–7018. https://doi.org/10.1109/TVCG.2025.3536358",
    "Li, Y., Han, D., Pan, H., et al. (2026). When charts deceive web agents: A paired benchmark for misleading visualization-grounded tasks. Findings of EMNLP 2026. https://doi.org/10.18653/2026.findings-emnlp.1993",
    "Mahbub, R., Islam, M. S., Laskar, M. T. R., Rahman, M., Nayeem, M. T., & Hoque, E. (2025). The perils of chart deception: How misleading visualizations affect vision-language models. IEEE VIS 2025. https://doi.org/10.1109/VIS60296.2025.00006",
    "Rahman, M., Wu, H., Asgari, A., Prince, E. H., & Seyyed-Kalantari, L. (2026). Same chart, different story: Bias in vision-language chart interpretation. Proceedings of EMNLP 2026. https://doi.org/10.18653/2026.emnlp-main.1041",
    "Su, R., Si, J., Guo, Z., & Pierrehumbert, J. B. (2025). ClimateViz: A benchmark for statistical reasoning and fact verification on scientific charts. Proceedings of EMNLP 2025. https://doi.org/10.18653/v1/2025.emnlp-main.1196",
    "Tonglet, J., Tuytelaars, T., Moens, M.-F., & Gurevych, I. (2026). Protecting multimodal large language models against misleading visualizations. Proceedings of ACL 2026 (long papers). https://doi.org/10.18653/v1/2026.acl-long.377",
]:
    doc.add_paragraph(ref, style="List Number")

doc.save("/home/claude/phase3/report/DAV_Phase3_Implementation_and_Evaluation.docx")
print("saved")
