from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import NamedVector
from rag.gen.qdrant_rag_test import CFG, CQ2

DOC = "1-case- Case Report_ Simple Nodular Cutaneous Leishmaniasis Caused by Autochthonous Leishmania (Mundinia) orientalis in an 18-Month-Old Girl_ The First Pediatric Case in Thailand and Literature Review"
q   = "Which therapy was used and what was the short-term outcome?"

enc = CQ2(CFG.RET_MODEL_ID)
qv  = enc.embed_texts([q])[0].tolist()
flt = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=DOC))])

c = QdrantClient(url=CFG.QDRANT_URL, api_key=CFG.QDRANT_API_KEY, prefer_grpc=False)

def query_any(qv, flt, limit=10):
    # 1) Try new API
    try:
        from qdrant_client.http.models import NearVector
        return c.query_points(
            collection_name=CFG.COLLECTION,
            query=NearVector(vector=qv, using="image"),
            limit=limit, filter=flt, with_payload=True
        )
    except Exception:
        # 2) Older query_points signature
        try:
            return c.query_points(
                collection_name=CFG.COLLECTION,
                query=qv, limit=limit, query_filter=flt, with_payload=True
            )
        except Exception:
            # 3) Legacy search
            return c.search(
                collection_name=CFG.COLLECTION,
                query_vector=NamedVector(name="image", vector=qv),
                limit=limit, query_filter=flt, with_payload=True
            )

res = query_any(qv, flt, limit=10)
print([(round(getattr(h, "score", 0.0), 4),
        (h.payload or {}).get("page_index")) for h in res])

ci = c.get_collection("leish_cases_pages")
print(ci.config.params.vectors)