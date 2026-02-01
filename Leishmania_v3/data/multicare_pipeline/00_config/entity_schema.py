"""
Entity schema definitions for the Leishmaniasis Knowledge Graph.

Based on Q1 journal standards (JAMIA, Artificial Intelligence in Medicine, 
Journal of Biomedical Informatics) and UMLS/SNOMED-CT ontology patterns.
"""
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class EntityType(Enum):
    """
    Entity types for medical knowledge graphs.
    
    Priority levels based on user requirements:
    - Priority 1: DISEASE, PATHOGEN (core classification)
    - Priority 2: SYMPTOM, SIGN, FINDING (clinical presentation)
    - Priority 3: TREATMENT, DRUG, PROCEDURE (clinical management)
    - Priority 4: ANATOMY, BODY_SITE (anatomical context)
    """
    # Priority 1: Core classification entities
    DISEASE = "Disease"
    PATHOGEN = "Pathogen"
    
    # Priority 2: Clinical presentation
    SYMPTOM = "Symptom"
    SIGN = "Sign"
    FINDING = "Finding"
    
    # Priority 3: Clinical management
    TREATMENT = "Treatment"
    DRUG = "Drug"
    PROCEDURE = "Procedure"
    TEST = "Test"
    
    # Priority 4: Anatomical context
    ANATOMY = "Anatomy"
    BODY_SITE = "BodySite"
    
    # Supporting entities
    CLINICAL_CASE = "ClinicalCase"
    IMAGE = "Image"


class RelationType(Enum):
    """
    Relation types connecting entities in the knowledge graph.
    """
    # Pathogen-Disease relations
    CAUSES = "CAUSES"  # Pathogen → Disease
    
    # Disease-Clinical relations
    HAS_SYMPTOM = "HAS_SYMPTOM"  # Disease → Symptom
    HAS_SIGN = "HAS_SIGN"  # Disease → Sign
    AFFECTS = "AFFECTS"  # Disease → Anatomy
    
    # Diagnostic relations
    DIAGNOSED_BY = "DIAGNOSED_BY"  # Disease → Procedure
    FINDING_IN = "FINDING_IN"  # Finding → Procedure
    INDICATES = "INDICATES"  # Finding → Disease
    
    # Treatment relations
    TREATED_WITH = "TREATED_WITH"  # Disease → Drug/Treatment
    ADMINISTERED_VIA = "ADMINISTERED_VIA"  # Drug → Procedure
    
    # Case relations
    CASE_HAS_DISEASE = "CASE_HAS_DISEASE"
    CASE_HAS_SYMPTOM = "CASE_HAS_SYMPTOM"
    CASE_HAS_TREATMENT = "CASE_HAS_TREATMENT"
    CASE_HAS_IMAGE = "CASE_HAS_IMAGE"
    
    # Image relations
    IMAGE_SHOWS = "IMAGE_SHOWS"  # Image → Finding/Anatomy


@dataclass
class Entity:
    """Represents a KG entity node."""
    id: str
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    synonyms: Optional[List[str]] = None
    external_ids: Optional[dict] = None  # UMLS CUI, SNOMED-CT, MeSH ID
    confidence: float = 1.0
    source: Optional[str] = None


@dataclass
class Relation:
    """Represents a KG relation edge."""
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float = 1.0
    evidence: Optional[str] = None
    source: Optional[str] = None


# =============================================================================
# LEISHMANIASIS-SPECIFIC ENTITY DEFINITIONS
# =============================================================================

# Disease entities (from your existing leishmaniasis_kg.json + MultiCaRe MeSH)
LEISHMANIA_DISEASES = {
    "VL": {
        "name": "Visceral Leishmaniasis",
        "synonyms": ["Kala-azar", "Kala azar", "VL", "Systemic Leishmaniasis"],
        "mesh_terms": ["Leishmaniasis, Visceral"]
    },
    "CL": {
        "name": "Cutaneous Leishmaniasis",
        "synonyms": ["CL", "Skin Leishmaniasis", "Oriental Sore", "Delhi Boil"],
        "mesh_terms": ["Leishmaniasis, Cutaneous"]
    },
    "MCL": {
        "name": "Mucocutaneous Leishmaniasis",
        "synonyms": ["MCL", "Mucosal Leishmaniasis", "Espundia"],
        "mesh_terms": ["Leishmaniasis, Mucocutaneous"]
    },
    "DCL": {
        "name": "Diffuse Cutaneous Leishmaniasis",
        "synonyms": ["DCL", "Anergic Cutaneous Leishmaniasis"],
        "mesh_terms": ["Leishmaniasis, Diffuse Cutaneous"]
    },
    "PKDL": {
        "name": "Post-Kala-Azar Dermal Leishmaniasis",
        "synonyms": ["PKDL", "Post-Kala-Azar Dermal"],
        "mesh_terms": []
    }
}

# Pathogen entities - EXPANDED based on dataset coverage check
LEISHMANIA_PATHOGENS = {
    "L_donovani": {
        "name": "Leishmania donovani",
        "synonyms": ["L. donovani", "L donovani"],
        "causes": ["VL", "PKDL"]
    },
    "L_infantum": {
        "name": "Leishmania infantum",
        "synonyms": ["L. infantum", "L infantum", "Leishmania chagasi", "L. chagasi", "L chagasi"],
        "causes": ["VL"]
    },
    "L_major": {
        "name": "Leishmania major",
        "synonyms": ["L. major", "L major"],
        "causes": ["CL"]
    },
    "L_tropica": {
        "name": "Leishmania tropica",
        "synonyms": ["L. tropica", "L tropica"],
        "causes": ["CL", "VL"]
    },
    "L_braziliensis": {
        "name": "Leishmania braziliensis",
        "synonyms": ["L. braziliensis", "L braziliensis", "Leishmania (Viannia) braziliensis"],
        "causes": ["CL", "MCL"]
    },
    "L_amazonensis": {
        "name": "Leishmania amazonensis",
        "synonyms": ["L. amazonensis", "L amazonensis"],
        "causes": ["CL", "DCL"]
    },
    # Additional species found in dataset
    "L_mexicana": {
        "name": "Leishmania mexicana",
        "synonyms": ["L. mexicana", "L mexicana"],
        "causes": ["CL"]
    },
    "L_guyanensis": {
        "name": "Leishmania guyanensis",
        "synonyms": ["L. guyanensis", "L guyanensis", "Leishmania (Viannia) guyanensis", "Viannia guyanensis"],
        "causes": ["CL"]
    },
    "L_panamensis": {
        "name": "Leishmania panamensis",
        "synonyms": ["L. panamensis", "L panamensis", "Leishmania (Viannia) panamensis"],
        "causes": ["CL"]
    },
    "L_aethiopica": {
        "name": "Leishmania aethiopica",
        "synonyms": ["L. aethiopica", "L aethiopica"],
        "causes": ["CL", "DCL"]
    },
    "L_peruviana": {
        "name": "Leishmania peruviana",
        "synonyms": ["L. peruviana", "L peruviana", "Leishmania (Viannia) peruviana"],
        "causes": ["CL"]
    },
    "L_naiffi": {
        "name": "Leishmania naiffi",
        "synonyms": ["L. naiffi", "L naiffi", "Leishmania (Viannia) naiffi"],
        "causes": ["CL"]
    },
    "L_lainsoni": {
        "name": "Leishmania lainsoni",
        "synonyms": ["L. lainsoni", "L lainsoni", "Leishmania (Viannia) lainsoni"],
        "causes": ["CL"]
    }
}

# Common symptoms and signs
LEISHMANIA_SYMPTOMS = [
    "fever", "weight loss", "fatigue", "malaise", "anorexia",
    "hepatomegaly", "splenomegaly", "hepatosplenomegaly",
    "lymphadenopathy", "pancytopenia", "anemia", "thrombocytopenia",
    "skin lesion", "ulcer", "nodule", "papule", "plaque",
    "nasal congestion", "epistaxis", "mucosal lesion"
]

# Common drugs and treatments - EXPANDED with synonyms and abbreviations
LEISHMANIA_DRUGS = [
    # Amphotericin formulations
    "amphotericin B", "liposomal amphotericin B", "AmBisome", "L-AmB",
    "amphotericin B deoxycholate", "conventional amphotericin",
    # Miltefosine
    "miltefosine", "Impavido",
    # Antimonials
    "sodium stibogluconate", "Pentostam", "SSG",
    "meglumine antimoniate", "Glucantime", "antimony", "antimonial",
    "pentavalent antimony", "pentavalent antimonial", "Sb5+",
    # Pentamidine
    "pentamidine", "pentamidine isethionate",
    # Azole antifungals
    "ketoconazole", "itraconazole", "fluconazole", "voriconazole",
    # Others
    "paromomycin", "aminosidine",
    "sitamaquine",
    "allopurinol",
    "thermotherapy", "cryotherapy",
    "intralesional"
]

# Common procedures and tests - EXPANDED based on dataset coverage
LEISHMANIA_PROCEDURES = [
    # Tissue sampling
    "bone marrow aspirate", "bone marrow aspiration", "bone marrow biopsy",
    "splenic aspirate", "splenic aspiration", "splenic puncture",
    "lymph node aspirate", "lymph node aspiration",
    "skin biopsy", "punch biopsy", "incisional biopsy",
    "skin scraping", "scraping",
    # Cytology/FNAC
    "FNAC", "fine needle aspiration", "fine needle aspiration cytology",
    "needle aspiration", "aspiration cytology",
    # Smear/microscopy
    "smear", "peripheral smear", "bone marrow smear",
    "Giemsa stain", "Giemsa-stained", "H&E stain", "Wright stain",
    "LD bodies", "Leishman-Donovan bodies", "amastigotes",
    # Molecular  
    "PCR", "real-time PCR", "qPCR", "polymerase chain reaction",
    # Culture
    "culture", "NNN medium", "promastigotes",
    # Serology
    "rK39 test", "rK39 rapid test", "rK39",
    "DAT", "direct agglutination test",
    "ELISA", "IFAT", "IFA",
    "Montenegro test", "leishmanin skin test", "LST",
    # Histology
    "immunohistochemistry", "IHC"
]

# Anatomical sites commonly affected
LEISHMANIA_ANATOMY = [
    "skin", "liver", "spleen", "bone marrow",
    "lymph node", "nasal mucosa", "oral mucosa",
    "face", "extremities", "trunk"
]
