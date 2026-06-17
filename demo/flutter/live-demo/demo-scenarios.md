# Demo Scenarios

Use these exact texts during rehearsal and supervisor presentation.

## Scenario 1: RAG-Supported

Input:

```text
Ulcerated plaque on the forearm after sandfly exposure with a smear showing amastigotes.
```

Expected state: `rag_supported`

Point out:

- Retrieved evidence cards are visible.
- Evidence includes title, diagnosis label, score, source case, and confirmatory flag.
- Gate support status is `supported`.
- Ranked differential is safe to show.

## Scenario 2: Insufficient Input

Input:

```text
Rash.
```

Expected state: `abstained`

Point out:

- No ranked differential is shown.
- Trigger includes missing required inputs.
- The app asks for safer next information.

## Scenario 3: Provisional Low-Support Input

Input:

```text
Chronic skin lesion with ulcerated border after travel to an endemic region.
```

Expected state: `provisional_parametric`

Point out:

- Retrieval support is weak.
- Output is explicitly model-only fallback.
- Evidence list is empty because the response is not evidence-grounded.

## Scenario 4: Out-Of-Scope Fallback

Input:

```text
Pigmented melanoma-like lesion with rapid growth and bleeding.
```

Expected state: `abstained`

Point out:

- Corpus-gap safety behavior.
- The leishmaniasis demo should not force an answer for an out-of-scope case.
