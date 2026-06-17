# Harness Engineering For RAG Diagnosis

This harness adapts the local OS harness-engineering outline to a medical RAG
research repository. The goal is not to make the model sound confident; the
goal is to make every answer, slide, and demo traceable to source artifacts and
bounded by explicit safety language.

## Reading Order

1. `00-overview/README.md`
2. `01-validation/README.md`
3. `02-agent-workflow/README.md`
4. `03-safety/README.md`

## Core Rule

Agent = model + harness. For this repository, the harness is the set of
validation scripts, provenance ledgers, branch/commit discipline, and
presentation boundaries that prevent research evidence from becoming an
overclaim.
