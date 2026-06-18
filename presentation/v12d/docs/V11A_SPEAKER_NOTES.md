# V12c Speaker Notes

Date: 2026-06-13

These notes support the V12c 30-minute route. They are intentionally shorter
than a script so the presenter can speak to the visible evidence rather than
read.

## Slides 1-4: Claim And Method

1. **Opening**
   State the bounded claim: multimodal RAG can make evidence use more traceable
   under resource constraints. It is not a diagnostic system and is not ready
   for clinical deployment.
2. **Clinical problem**
   Leishmaniasis decisions combine lesion appearance, symptoms, exposure,
   geography, and confirmatory tests. That is why image classification alone is
   the wrong abstraction.
3. **Research questions**
   Connect the four questions into one thread: retrieval quality, multimodal
   evidence, model dependence, and practical local deployment.
4. **Architecture**
   Walk left to right through text and image-linked retrieval, fusion, the
   uncertainty gate, and the auditable final route.

## Slides 5-11: Main Findings

5. **Model-dependent RAG effect**
   Lead with the result visible on the chart: retrieval is not uniformly
   beneficial. The generator and context quality determine whether it helps.
6. **Corpus imbalance**
   Explain that an imbalanced disease-heavy corpus can damage specificity.
   Retrieval can amplify the corpus prior rather than correct it.
7. **Retriever ceiling**
   A generator cannot reliably use evidence that the retriever does not return.
   This is a system result, not only a language-model result.
8. **Image evidence**
   Images add context but also create hallucination and overinterpretation
   risks. Never describe the image as standalone diagnostic proof.
9. **Small-model practicality**
   Emphasize feasibility on constrained local hardware, then separate
   feasibility from clinical validity.
10. **Limitations**
    Present limitations as explicit boundaries: silver labels, corpus size and
    mix, retrieval errors, image interpretation risk, and no clinical trial.
11. **Conclusion**
    Close the first arc: feasible and traceable under constraints, with
    model-dependent benefits and no deployment claim.

## Slides 12-14: Evidence Map

12. **Result map**
    Use this slide to connect each headline claim to the supporting result
    family. Do not spend time on every box.
13. **Mechanism audit**
    Explain how clean-context and route checks help distinguish a real retrieval
    effect from formatting or routing artifacts.
14. **Sensitivity and practicality**
    Summarize modality, latency, and GPU sensitivity. Transition from aggregate
    evidence to individual case inspection.

## Slides 15-19: Case Inspection

15. **Inspection setup**
    Say all four boundaries explicitly:
    - Three illustrative queries from the 56-case held-out set, not an accuracy
      sample.
    - IDs verified absent from the official 121-case Tier A+B experimental
      retrieval corpus.
    - Targets are silver references, not clinician-adjudicated ground truth.
    - Saved outputs used a separate small local defense demo KB.
16. **MCL concordance**
    The model rank-1 type matches the MCL silver reference. Treat this as a
    readable concordant trace, not proof of accuracy or diagnosis.
17. **PKDL subtype mismatch**
    The model retains the leishmaniasis family but shifts PKDL toward MCL. This
    is the clearest example of the subtype-resolution limit.
18. **Non-leish specificity**
    The rank-1 type remains outside leishmaniasis. Use it to illustrate the
    specificity boundary, not to claim image-based exclusion.
19. **Close**
    Restate the bounded contribution: an inspectable local multimodal RAG
    pipeline with clear failure modes, not a deployment-ready clinical tool.

## Slides 20-25: Appendix

- Open slides 20-22 only when asked to inspect the complete blinded input.
- Point out that diagnosis, title, abstract, captions, and confirmatory findings
  were withheld.
- Do not restart the case narrative; answer the specific provenance or input
  question and return to slide 19.
- Open slides 23-25 only when asked about RAG versus no-RAG behavior for the
  same selected examples.
- State that the comparison comes from official Gemma 4 experiment-pipeline
  runs, not the failed demo-backend no-RAG attempt.
- Do not turn the three backup examples into an aggregate accuracy claim.

## Likely Committee Questions

**Are these labels ground truth?**  
No. They are silver reference labels produced by the thesis pipeline and are
not clinician-adjudicated gold labels.

**Were the examples retrieved from the same 121-case corpus?**  
The selected query IDs were verified absent from the official 121-case
experimental retrieval corpus. The preserved defense outputs used a separate
small local demo KB for runtime support.

**Do three cases demonstrate accuracy?**  
No. They are illustrative traces. Aggregate thesis metrics remain the
benchmark.

**Why show images if the system is not an image diagnostic tool?**  
Images are contextual evidence in a multimodal workflow. They are not treated
as standalone diagnostic proof.

**Was V12c a new backend model run?**  
No. V12c preserves the safer v11a/v12b case story for the spoken deck. The
RAG/no-RAG backup appendix uses existing official Gemma 4 experiment-pipeline
artifacts.

**What would be required before deployment?**  
Clinician-adjudicated labels, external validation, broader and balanced data,
calibrated uncertainty and abstention, workflow testing, and prospective safety
evaluation.
