# V7 30-Minute Colloquium Run Of Show

Date: 2026-06-13

## Route

Target spoken time: exactly 30:00, excluding Q&A.

- Slides 1-18: spoken colloquium path.
- Slide 19: close and Q&A bridge.
- Slides 20-22: appendix-only evidence for supervisor questions.

## Timing Table

| Time | Slide | Purpose | Checkpoint |
| --- | ---: | --- | --- |
| 0:00-1:00 | 1 | Opening, thesis claim, safety boundary | Leave slide 1 by 1:00 |
| 1:00-2:30 | 2 | Problem: leishmaniasis diagnosis is multimodal | Finish motivation by 2:30 |
| 2:30-4:00 | 3 | Research questions and evaluation thread | Finish questions by 4:00 |
| 4:00-6:00 | 4 | Two-lane retrieval architecture | Finish method overview by 6:00 |
| 6:00-8:00 | 5 | Main RAG result and model dependence | State the main result clearly |
| 8:00-10:00 | 6 | Corpus imbalance and specificity risk | Explain why retrieval can hurt |
| 10:00-12:00 | 7 | Retriever quality ceiling | Connect retrieval to answer quality |
| 12:00-14:00 | 8 | Image evidence: useful but risky | Reinforce image-as-context boundary |
| 14:00-16:00 | 9 | Small-model practicality | Tie feasibility to GPU constraints |
| 16:00-18:00 | 10 | Limitations | Make limitations sound designed-in |
| 18:00-19:30 | 11 | Core conclusion | Close the first arc by 19:30 |
| 19:30-21:00 | 12 | Evidence map | Transition from claim to evidence |
| 21:00-22:30 | 13 | Clean-context and route audit | Explain the RAG effect mechanism |
| 22:30-24:00 | 14 | Sensitivity and practicality | Finish core evidence by 24:00 |
| 24:00-25:30 | 15 | Held-out case-analysis setup | Start case walkthrough by 25:30 |
| 25:30-26:45 | 16 | Case 1: held-out MCL match | Keep it to one success-style point |
| 26:45-28:00 | 17 | Case 2: PKDL subtype limitation | Explain subtype boundary clearly |
| 28:00-29:15 | 18 | Case 3: label-conflict stress test | Explain why disputed labels need audit |
| 29:15-30:00 | 19 | Final claim and Q&A handoff | Stop at 30:00 |

## Transition Lines

Opening to problem:
"The thesis is not that an image classifier can diagnose leishmaniasis. The
claim is narrower: under resource constraints, multimodal RAG can make evidence
use more traceable, but only within clear safety boundaries."

Problem to method:
"Because the clinical problem mixes text, images, exposure history, and
confirmatory findings, the system design has to preserve evidence routes rather
than collapse everything into a single prediction."

Method to results:
"With that architecture in mind, the first question is whether retrieval helps.
The answer is model-dependent, and that is one of the main findings."

Core results to live evaluation:
"The aggregate slides explain the mechanism. The next four slides show how the
same system behaves on three selected queries from the 56-case held-out
evaluation set. They are not clinical validation, but they let us inspect the
input-output-evidence path on cases absent from the clinical retrieval corpus."

Evaluation to close:
"These cases are not clinical validation. They are functional evidence: the
pipeline runs, retrieves, reasons, and exposes exactly where it is weak."

Close:
"The final claim is therefore deliberately bounded: feasible and traceable under
constraints; not deployment-ready."

## Demo Handling

The evaluation segment on slides 15-18 can support a short live or recorded
demo, but the deck should remain complete without it.

- Preferred live path: show slide 15, run or describe the `PMC7516301_01`
  case, then return to slide 16 for the already rendered result.
- Time budget: 3-5 minutes inside the slide 15-18 segment. If live inference is
  slow after 90 seconds, stop waiting and use the rendered result.
- Fallback line: "I will use the saved run here because the scientific point is
  the traceable input-output-evidence path, not the latency of this particular
  live execution."
- Boundary line: "This is a blinded functional evaluation, not clinical
  validation and not a diagnosis from the image alone."
- Provenance line: "These examples come from the 56-case held-out evaluation
  set and were checked as absent from the 121-case clinical retrieval corpus."

## Appendix Map

- Slide 20: full blinded input for `PMC7516301_01`.
- Slide 21: full blinded input for `PMC7456484_01`.
- Slide 22: full blinded input for `PMC10026180_04`.

Use appendix slides only when a supervisor asks to inspect the full case input,
check whether diagnosis/title/abstract/captions were withheld, confirm that the
case came from the held-out evaluation set, or compare the compact slide
summary against the original evaluation JSON. Do not present three cases as a
replacement for the thesis aggregate benchmark metrics.

## Reference Principles

- MIT CommLab slide-design guidance supports slides that amplify the talk
  rather than replace it.
- MIT CommLab technical-demo guidance supports demos with context, visible
  evidence, pacing, risk management, and fallback handling.
- Stanford oral-exam policy supports treating the event as both a public
  research explanation and a rigorous committee-questioning setting.
- Stanford, Harvard, and UCLA dissertation/thesis pages reinforce disciplined,
  reviewable final materials with clear institutional expectations.
