# rag/test/quick_check.py
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import NamedVector
from rag.reranking.med4b_qdrant_bge import CFG, CQ2, _qdrant_search

DOC = "1-case- Case Report_ Simple Nodular Cutaneous Leishmaniasis Caused by Autochthonous Leishmania (Mundinia) orientalis in an 18-Month-Old Girl_ The First Pediatric Case in Thailand and Literature Review"
q   = "Which therapy was used and what was the short-term outcome?"

enc = CQ2(CFG.RET_MODEL_ID)
qv  = enc.embed_texts([q])[0]
flt = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=DOC))])

c = QdrantClient(url=CFG.QDRANT_URL, api_key=CFG.QDRANT_API_KEY, prefer_grpc=False)
res = _qdrant_search(c, qv, 10, flt, None)
pts = getattr(res, "points", []) or res  # tolerate both return types
print([(round(getattr(h, "score", 0.0), 4),
        (getattr(h, "payload", {}) or {}).get("page_index")) for h in pts])
