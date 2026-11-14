# Phase 1 thesis format and submission requirements

Recorded from the user's supplied PG Academics email and local template on 29 October 2026. The email is the source for submission requirements; the files below are format references containing example content, not project evidence.

## Authoritative local format references

Template directory: `E:/E-Mtech/Project/MWP-Project Report Template - Cyber Security-2026/`

- `report.tex`: master LaTeX source and front matter.
- `chapter1.tex`, `chapter2.tex`: example chapter syntax, not a two-chapter limit.
- `bib.bib`: sample BibTeX entries to replace with verified project references.
- `logo2.jpg`: supplied institute logo.
- `MWP_Project_Report_Template___Cyber_Security_June2025.pdf`: compiled sample reference.

Preserve the supplied originals. Draft the actual thesis in a dedicated local LaTeX project based on this template when authoring is requested. Do not treat sample mathematics, Cyber Security wording, dates, or bibliography as thesis content.

## Format observed in the supplied sources

- LaTeX report class with 12pt text; source specifies `a4wide` as a class option.
- Source sets baseline stretch to 1.5.
- Front matter order: title page, declaration, supervisor certificate, abstract, table of contents, list of figures, list of tables.
- Roman numbering in front matter and Arabic numbering restarting at 1 for main chapters.
- Separate chapter source files included from the master file.
- BibTeX bibliography using `plain` style in the sample.
- Preserve the institute identity and logo proportions.
- Replace title, candidate details, supervisor details, programme wording, and dates consistently.

Observed discrepancy: the supplied PDF has 612 x 792 point pages (US Letter), despite the source's `a4wide` option. Do not assume that option sets A4 paper or margins. Resolve the required paper size/margins against institute/mentor guidance before final layout; do not silently invent dimensions. Source and PDF also contain different old dates.

## Submission requirements supplied by PG Academics

- Prepare the report in LaTeX using the shared institutional format.
- Include digital signatures of both candidate and supervisor. Actual signatures must come from the respective signatories; placeholders do not satisfy submission requirements.
- Submit the plagiarism report together with the thesis. Obtain it through the supervisor.
- Plagiarism percentage must be strictly less than 20%.
- Avoid plagiarism and self-plagiarism, including in prose, images, equations, and code.
- Cite reproduced journal images and obtain suitable author/publisher permission as required by the email.
- Discuss chapters and content with the supervisor and obtain approval.
- Incorporate all internal-panel suggestions before final resubmission.
- Draft deadline has no extension according to the supplied email.

LMS submission location supplied by the user: https://lmspg24.iiitkottayam.ac.in/course/section.php?id=2355
The LMS has not been accessed or verified in this task.

## Dates and dependencies (Asia/Kolkata / IST)

| Milestone | Date/time | Source |
| --- | --- | --- |
| Mentor receives complete first draft | Sunday, 1 November 2026; time not specified | User's planning deadline |
| Draft thesis and plagiarism report submitted through LMS | 5 November 2026, 11:59 PM IST | Supplied PG Academics email |
| Internal panel review/comments | 13 November 2026 | Supplied PG Academics email |
| Final thesis resubmission incorporating feedback | 18 November 2026; time not specified | Supplied PG Academics email |
| Phase 1 final review and demonstration | 20 November 2026; time not specified | User's review schedule |

Plan mentor review, similarity checking, required revisions, and both signatures before the 5 November submission. Reserve 13-17 November for panel corrections and final thesis preparation. Thesis results and evidence must be ready before 18 November; 19 November is available for demonstration rehearsal, not new thesis experiments. Confirm any final-submission time on LMS when available.

## Internal evaluation criteria

1. Definition of project scope and goals.
2. Literature survey.
3. Presentation of methods.
4. Analysis and results.
5. Relevant and recent scientific references.
6. Overall organisation.

## Working chapter outline (proposed, pending supervisor approval)

1. Introduction: problem, motivation, scope, objectives, research questions, Phase 1 boundary.
2. Literature survey: query chunking, intent classification, transformers, query refinement, research gap.
3. Dataset and methodology: ESCI, annotation taxonomy/guidelines, reviewed labels, splits, architecture, evaluation protocol.
4. Phase 1 implementation: rule-based chunking, Gemini, transformer experiments, intent baselines, application and cloud deployment status.
5. Experiments and discussion: verified results, class support, error analysis, reproducibility, limitations.
6. Conclusion and future work: established Phase 1 contributions and planned Phase 2 retrieval work.

Include only actually completed work as implemented/results. Mark transformer and cloud work as planned until verified. Never fabricate results, references, approval, signatures, or a similarity score. Exact official programme wording and supervisor display name should be confirmed before final front matter.
