# Supervisor Demo Script

## Opening

"Today I will show a hybrid thesis demo. The slides summarize the research pipeline and results. The live app demonstrates the product-facing contract: input capture, evidence retrieval, uncertainty gating, and safe abstention."

## Important Boundary

"The live backend uses deterministic demo logic for reliability. It does not run real clinical inference. The thesis evidence and model-run provenance are in the RTX6000 experiment artifacts."

## One-Sentence Architecture

"The Flutter app sends a patient description and optional image to a local Python backend; the backend retrieves from a small curated leishmaniasis knowledge base, applies uncertainty gates, and returns either an evidence-supported response, a provisional fallback, or abstention."

## Demo Case 1: Supported

Paste:

```text
Ulcerated plaque on the forearm after sandfly exposure with a smear showing amastigotes.
```

Say:

"This case has enough clinical and confirmatory detail for a supported path. The important parts are the retrieved evidence cards, top score, support status, and disclaimer."

## Demo Case 2: Insufficient

Paste:

```text
Rash.
```

Say:

"This input is deliberately too vague. The correct behavior is not to force a ranked differential. The gate blocks the output and asks for safer next inputs."

## Demo Case 3: Provisional

Paste:

```text
Chronic skin lesion with ulcerated border after travel to an endemic region.
```

Say:

"This case is plausible but not strongly evidence-grounded. The backend allows a provisional model-only fallback and does not claim retrieved support."

## Closing

"The thesis result is not that RAG always improves diagnosis support. The result is a traceable evaluation framework showing model-dependent benefits and risks. The live app makes the same safety contract understandable in a supervisor demo."

## Expected Questions

### Is this a diagnosis system?

No. It is decision-support infrastructure and a supervisor demo. It includes abstention and clinician-review boundaries.

### Is the live model real?

No. The live path is deterministic for reliability. The real evaluation runs are represented by the thesis and RTX6000 artifacts.

### Why not run Gemma4 live?

The thesis reports Gemma4 local generation around 46-59 seconds per case, which is too slow and fragile for a dependable live supervisor demo.
