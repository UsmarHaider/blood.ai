# utils.py

def get_disease_description(disease: str) -> str:
    """Get a general description of the blood disease or genetic marker."""
    descriptions = {
        "NPM1": "NPM1 is a genetic mutation commonly found in acute myeloid leukemia (AML). "
                "It affects the nucleophosmin protein and is generally associated with a more "
                "favorable prognosis compared to some other genetic markers in AML.",

        "PML_RARA": "PML-RARA is a fusion gene associated with acute promyelocytic leukemia (APL), "
                    "a subtype of acute myeloid leukemia. This genetic abnormality is caused by a "
                    "translocation between chromosomes 15 and 17, and is responsive to targeted therapies.",

        "RUNX1_RUNX1T1": "RUNX1-RUNX1T1 (previously known as AML1-ETO) is a fusion gene resulting "
                         "from a translocation between chromosomes 8 and 21. It is associated with a "
                         "specific subtype of acute myeloid leukemia (AML) that generally has a favorable prognosis.",

        "control": "This indicates a sample considered normal or negative for the specific genetic markers "
                   "being tested for in this context (e.g., NPM1, PML_RARA, RUNX1_RUNX1T1). It does not rule out other conditions.",

        # Add descriptions for cell types if needed, or keep them separate
        "ig": "Immature Granulocytes (IG) are young white blood cells, precursors to neutrophils, eosinophils, and basophils. Increased numbers can indicate infection, inflammation, or other conditions.",
        "lymphocyte": "Lymphocytes are a type of white blood cell crucial for the immune system, including B cells, T cells, and NK cells. They fight infections and cancer.",
        "monocyte": "Monocytes are the largest type of white blood cell. They circulate in the blood before migrating into tissues, where they differentiate into macrophages or dendritic cells, playing roles in phagocytosis and antigen presentation.",
        "neutrophil": "Neutrophils are the most common type of granulocyte and white blood cell, forming an essential part of the innate immune system. They are typically the first responders to bacterial infections.",
        "platelet": "Platelets (thrombocytes) are small, irregular-shaped cell fragments that circulate in the blood and are involved in hemostasis, leading to the formation of blood clots.",
    }

    return descriptions.get(disease, f"General information about '{disease}' is not available in this quick reference.")