**Model Output Assessment — Case 2 / PKDL**

Gemma 4 RAG performs reasonably well at the **disease-family level**, but it misses the more specific subtype.

The model ranks **Mucocutaneous Leishmaniasis (MCL)** as the top supportive consideration, with **Cutaneous Leishmaniasis (CL)** second and non-leishmaniasis mimics third. This is not an irrational answer, because the case includes chronic facial plaques, nasal mucosal extension, eyelid damage/scarring, and a history of prior visceral leishmaniasis. These features can push the model toward a leishmaniasis-related diagnosis.

However, if the silver reference label is **Post-Kala-Azar Dermal Leishmaniasis (PKDL)**, then this output should be interpreted as a **subtype-resolution failure**. The strongest clinical pattern in the input is:

**prior treated VL → later chronic facial plaques/papules → WHO consultant suspected PKDL**

Gemma 4 recognizes the broader leishmaniasis family, but it overweights the nasal mucosal involvement and retrieves MCL/CL/VL support instead of explicit PKDL support. The retrieval trace explains the error: the local demo retrieved **VL, MCL, and CL chunks**, but no clear PKDL-specific evidence. Therefore, the generation is pulled toward MCL.

The model’s **medium confidence** is appropriate because the case is ambiguous and previous biopsies failed to confirm leishmaniasis. It is also good that the model keeps lupus/dermatitis as a non-leishmaniasis differential.

**Conclusion:**
This case should be presented as a **family-level success but subtype-level mismatch**. Gemma 4 correctly stays within the leishmaniasis family, but it fails to identify PKDL as the best subtype. This is useful evidence for the thesis because it shows that RAG can support the correct disease family while still confusing clinically related subtypes when retrieval does not provide the right subtype-specific evidence.
