# V5 / V7 Layout Recheck

Date: 2026-06-13

## Result

V7 is more layout-consistent than V5 and should remain the presentation file.

Structured comparison:

| Check | V5 | V7 |
| --- | ---: | ---: |
| Slide count | 21 | 22 |
| Slide size | 10 x 5.625 in | 10 x 5.625 in |
| Dedicated Thank You / Questions slide | no | yes, slide 19 |
| Top-right page badge geometry | mixed | consistent |
| Legacy bottom-right page numbers | 7 slides | 0 slides |
| Rendered final-slide set | slides 15-21 | slides 1-22 + contact sheet |

V7 uses a single top-right page-badge geometry on content and appendix slides.
The title slide and the Thank You / Questions slide are intentionally
unnumbered. This is cleaner than V5, where the generated evaluation/appendix
slides still carried bottom-right slide numbers.

## Logo Placement

The VGU/h_da lockup appears on the title slide and the closing slide only. That
is a standard, conservative academic-deck pattern: establish institutional
identity at the beginning, return to it at the end, and avoid adding logo
clutter to every technical content slide.

This is a style standard rather than a universal rule. If the program provides
an official presentation template with different placement rules, that template
should override this deck-level choice. In the absence of such a rule, the
current V7 placement is appropriate and restrained.

## Visual Recheck

Fresh PNG renders were regenerated for all 22 slides in `qa/final_render/`.
Slides 15-22 were inspected after the latest wording changes:

- Slide 15: retrieval-corpus provenance line is readable.
- Slides 16-18: post-hoc QA attribution is readable and no longer clipped.
- Slides 20-22: appendix text remains legible for Q&A inspection.
- Contact sheet confirms that slides 1-14 preserve the standardized V7 visual
  system.
