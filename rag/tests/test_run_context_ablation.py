from pipeline.run_context_ablation import Variant, _build_variant_retrieval


def test_build_variant_retrieval_filters_qrels_without_pre_slicing_context_k():
    base_rows = [
        {
            "qid": "case-1::Q1_Q3_multimodal_diagnosis",
            "contexts": [
                {"doc_id": "doc-1"},
                {"doc_id": "doc-2"},
                {"doc_id": "doc-3"},
                {"doc_id": "doc-4"},
            ],
        }
    ]
    qrels_case_map = {"case-1": {"doc-1", "doc-2", "doc-3"}}
    variant = Variant(name="k2_qrels", context_k=2, constraint="qrels_only")

    variant_rows = _build_variant_retrieval(base_rows, qrels_case_map, variant)

    assert [ctx["doc_id"] for ctx in variant_rows[0]["contexts"]] == ["doc-1", "doc-2", "doc-3"]
