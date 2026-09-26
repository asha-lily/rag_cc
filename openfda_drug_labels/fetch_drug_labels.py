"""Fetch one openFDA drug label per product listed in the products CSV.

CSV columns (config.products_path):
    id            unique name for the product, used as the output filename
    ingredients   one or more ingredients separated by ';'
    product_type  HUMAN PRESCRIPTION DRUG or HUMAN OTC DRUG
    rxcui         optional RxNorm SCD to target a specific strength and form;
                  required when two rows share the same ingredients
    note          optional free text (ignored by this script)

Label selection: prefer a reference label (NDA, or BLA for biologics),
otherwise the most recently updated matching label.

Requires: uv add requests
Set OPENFDA_API_KEY in your .env file (optional, but raises the daily limit).

Usage:
    uv run python -m fetch_drug_labels
"""

import csv
import json
import logging
import os
import time

import requests
from dotenv import load_dotenv

from config import DrugLabelsConfig

load_dotenv()
log = logging.getLogger(__name__)
API_KEY = os.getenv("OPENFDA_API_KEY")
REFERENCE_PREFIXES = ("NDA", "BLA")


def load_products(config: DrugLabelsConfig) -> list[dict]:
    with config.products_path.open(newline="") as f:
        products = []
        for row in csv.DictReader(f):
            if not (row.get("id") or "").strip():
                continue
            row = {k: (v or "").strip() for k, v in row.items()}
            row["ingredients"] = [
                p.strip().lower() for p in row["ingredients"].split(";") if p.strip()
            ]
            products.append(row)
    return products


def build_query(product: dict) -> str:
    if product["rxcui"]:
        terms = [f'openfda.rxcui:"{product["rxcui"]}"']
    else:
        terms = [f'openfda.generic_name:"{name}"' for name in product["ingredients"]]
    terms.append(f'openfda.product_type:"{product["product_type"]}"')
    return " AND ".join(terms)


def fetch_labels(
    session: requests.Session, config: DrugLabelsConfig, query: str
) -> list[dict]:
    """Return up to config.page_size labels matching the query ([] if none match)."""
    params = {"search": query, "limit": config.page_size}
    if API_KEY:
        params["api_key"] = API_KEY
    resp = session.get(config.api_url, params=params, timeout=30)
    if resp.status_code == 404:  # openFDA returns 404 when nothing matches
        return []
    resp.raise_for_status()
    return resp.json().get("results", [])


def matches_ingredients(label: dict, ingredients: list[str]) -> bool:
    """True if the label has exactly these active ingredients (any salt form)."""
    substances = [s.lower() for s in label.get("openfda", {}).get("substance_name", [])]
    if len(substances) != len(ingredients):
        return False
    return all(any(ing in s for s in substances) for ing in ingredients)


def is_reference_label(label: dict) -> bool:
    app_numbers = label.get("openfda", {}).get("application_number", [])
    return any(num.startswith(REFERENCE_PREFIXES) for num in app_numbers)


def choose_label(labels: list[dict]) -> dict:
    """Prefer reference (NDA/BLA) labels, then the most recently updated."""
    return max(
        labels, key=lambda lbl: (is_reference_label(lbl), lbl.get("effective_time", ""))
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = DrugLabelsConfig()
    config.raw_labels_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    with requests.Session() as session:
        for product in load_products(config):
            out_path = config.raw_labels_dir / f"{product['id']}.json"
            if out_path.exists():  # makes re-runs resumable
                log.info("Already fetched: %s", product["id"])
                continue

            candidates = fetch_labels(session, config, build_query(product))
            time.sleep(config.request_interval_s)

            matching = [
                lbl
                for lbl in candidates
                if matches_ingredients(lbl, product["ingredients"])
            ]
            entry = {
                "id": product["id"],
                "ingredients": product["ingredients"],
                "product_type": product["product_type"],
                "rxcui_requested": product["rxcui"] or None,
                "n_candidates": len(candidates),
                "n_matching": len(matching),
            }

            if not matching:
                hint = " (try the related SBD rxcui)" if product["rxcui"] else ""
                log.warning("No matching label for %s%s", product["id"], hint)
                manifest.append({**entry, "status": "not_found"})
                continue

            best = choose_label(matching)
            out_path.write_text(json.dumps(best, indent=2))

            openfda = best.get("openfda", {})
            manifest.append(
                {
                    **entry,
                    "status": "ok",
                    "file": out_path.name,
                    "set_id": best.get("set_id"),
                    "brand_name": openfda.get("brand_name"),
                    "manufacturer_name": openfda.get("manufacturer_name"),
                    "application_number": openfda.get("application_number"),
                    "is_reference_label": is_reference_label(best),
                    "rxcui": openfda.get("rxcui"),
                    "effective_time": best.get("effective_time"),
                }
            )
            log.info(
                "Saved %s (%d matching of %d)",
                product["id"],
                len(matching),
                len(candidates),
            )

    (config.raw_labels_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_ok = sum(m["status"] == "ok" for m in manifest)
    log.info("Done: %d saved, %d not found", n_ok, len(manifest) - n_ok)


if __name__ == "__main__":
    main()
