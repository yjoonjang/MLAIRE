"""Build a Croissant file with Responsible-AI (RAI) fields for an MLAIRE dataset repo.

Hugging Face serves an auto-generated Croissant description for every dataset
repo, but without the RAI fields that the NeurIPS dataset checklist looks for
(e.g. ``rai:hasSyntheticData``). This script downloads that description and adds
the RAI fields below.

Usage:
    python scripts/make_croissant.py \\
        --repo <hf-user>/mlaire-mlqa --dataset mlqa \\
        --output croissant/mlaire-mlqa.json
"""

import argparse
import json
import urllib.request
from pathlib import Path

CROISSANT_URL = "https://huggingface.co/api/datasets/{repo}/croissant"
RAI_CONTEXT = "http://mlcommons.org/croissant/RAI/"
RAI_SPEC = "http://mlcommons.org/croissant/RAI/1.0"

SOURCES = {
    "mlqa": ("MLQA (Lewis et al., 2020)", "CC BY-SA 3.0"),
    "xquad": ("XQuAD (Artetxe et al., 2020)", "CC BY-SA 4.0"),
    "belebele": ("Belebele (Bandarkar et al., 2024), derived from FLORES-200", "CC BY-SA 4.0"),
}


def rai_fields(dataset: str) -> dict:
    source, license_ = SOURCES[dataset]
    return {
        "rai:hasSyntheticData": False,
        "rai:dataCollection": (
            f"Reorganized from the public {source} release without new annotation. "
            "Passages expressing the same content in different languages are grouped into "
            "content groups; every query is linked to its content group, and each passage "
            "and query keeps its language label."
        ),
        "rai:dataCollectionType": ["Secondary Data Analysis"],
        "rai:dataPreprocessingProtocol": (
            "Passages and queries are taken verbatim from the source release and exported in "
            "BEIR format (corpus, queries, qrels). Relevance is binary group membership."
        ),
        "rai:dataUseCases": (
            "Evaluating multilingual retrievers on cross-lingual semantic retrieval and on "
            "query-language preference (LPR, Lang-nDCG, top-1 4-way decomposition)."
        ),
        "rai:dataLimitations": (
            "QA-style parallel passages may not represent open-domain mixed-language corpora. "
            "Language coverage is uneven across families, scripts, and resource levels. "
            "Query-language preference is not a universal utility measure."
        ),
        "rai:dataBiases": (
            "High-resource languages and translated benchmark content may be overrepresented "
            "relative to naturally occurring multilingual search."
        ),
        "rai:personalSensitiveInformation": (
            "None added. Content is inherited from the public source dataset."
        ),
        "rai:dataSocialImpact": (
            "Intended as a diagnostic for whether retrievers return evidence in the query "
            "language. Optimizing language preference in isolation may hide more complete "
            f"cross-lingual evidence. Source license: {license_}; follow its attribution and "
            "share-alike terms."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face dataset repo id")
    ap.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with urllib.request.urlopen(CROISSANT_URL.format(repo=args.repo), timeout=60) as r:
        doc = json.load(r)

    doc["@context"]["rai"] = RAI_CONTEXT
    conforms = doc.get("conformsTo", [])
    conforms = [conforms] if isinstance(conforms, str) else list(conforms)
    doc["conformsTo"] = conforms + [RAI_SPEC]
    doc["license"] = SOURCES[args.dataset][1]
    doc.update(rai_fields(args.dataset))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
