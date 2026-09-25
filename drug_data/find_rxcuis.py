"""List RxNorm SCD candidates for each drug, with how many openFDA labels use each.

An SCD (Semantic Clinical Drug) is RxNorm's "ingredient + strength + dose form"
concept, e.g. "acetaminophen 500 MG Oral Tablet".

Usage:
    # Look up every ingredient in drug_data/drug_products.csv
    uv run python -m find_rxcuis

    # Or look up specific drugs (use ';' for combination products)
    uv run python -m find_rxcuis acetaminophen "acetaminophen; hydrocodone"

    # Optionally keep only one dose form
    uv run python -m find_rxcuis --form "oral tablet" metformin

Writes config.candidates_path, sorted so that for each drug the SCDs referenced
by the most openFDA labels come first. Within one ingredient, strengths made by
many manufacturers are usually the common ones, but treat this as a hint only.
"""

import argparse
import csv
import logging
import os
import time

import requests
from dotenv import load_dotenv

from config import DrugLabelsConfig

load_dotenv()
log = logging.getLogger(__name__)
API_KEY = os.getenv("OPENFDA_API_KEY")


def parse_ingredients(drug: str) -> list[str]:
    return [part.strip().lower() for part in drug.split(";") if part.strip()]


def drugs_from_products_csv(config: DrugLabelsConfig) -> list[str]:
    """Unique ingredient strings from the products CSV, in file order."""
    with config.products_path.open(newline="") as f:
        drugs = [
            row["ingredients"].strip()
            for row in csv.DictReader(f)
            if row.get("ingredients")
        ]
    return list(dict.fromkeys(drugs))


def get_scds(
    session: requests.Session, config: DrugLabelsConfig, ingredients: list[str]
) -> list[dict]:
    """Return SCD concepts containing exactly these ingredients."""
    resp = session.get(
        f"{config.rxnav_url}/drugs.json", params={"name": ingredients[0]}, timeout=30
    )
    resp.raise_for_status()
    groups = resp.json().get("drugGroup", {}).get("conceptGroup") or []

    scds = []
    for group in groups:
        if group.get("tty") != "SCD":
            continue
        for concept in group.get("conceptProperties", []):
            name = concept["name"].lower()
            # Components are separated by " / " (spaced), e.g.
            # "acetaminophen 325 MG / hydrocodone bitartrate 5 MG Oral Tablet".
            # Unspaced slashes are units, e.g. "160 MG/5ML".
            n_components = name.count(" / ") + 1
            if n_components == len(ingredients) and all(
                ing in name for ing in ingredients
            ):
                scds.append({"rxcui": concept["rxcui"], "name": concept["name"]})
    return scds


def count_openfda_labels(
    session: requests.Session, config: DrugLabelsConfig, rxcui: str
) -> int:
    params = {"search": f'openfda.rxcui:"{rxcui}"', "limit": 1}
    if API_KEY:
        params["api_key"] = API_KEY
    resp = session.get(config.api_url, params=params, timeout=30)
    if resp.status_code == 404:  # openFDA returns 404 when nothing matches
        return 0
    resp.raise_for_status()
    return resp.json()["meta"]["results"]["total"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "drugs", nargs="*", help="Drugs to look up (default: all in the products CSV)"
    )
    parser.add_argument(
        "--form",
        help="Only keep SCDs whose name contains this text, e.g. 'oral tablet'",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = DrugLabelsConfig()
    drugs = args.drugs or drugs_from_products_csv(config)

    rows = []
    with requests.Session() as session:
        for drug in drugs:
            ingredients = parse_ingredients(drug)
            scds = get_scds(session, config, ingredients)
            if args.form:
                scds = [s for s in scds if args.form.lower() in s["name"].lower()]
            log.info("%s: %d SCD candidates", drug, len(scds))

            drug_rows = []
            for scd in scds:
                n_labels = count_openfda_labels(session, config, scd["rxcui"])
                time.sleep(config.request_interval_s)
                drug_rows.append({"drug": drug, **scd, "n_openfda_labels": n_labels})

            drug_rows.sort(key=lambda r: r["n_openfda_labels"], reverse=True)
            for row in drug_rows[:5]:
                log.info(
                    "    %6d labels  %-10s %s",
                    row["n_openfda_labels"],
                    row["rxcui"],
                    row["name"],
                )
            rows.extend(drug_rows)

    config.candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with config.candidates_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["drug", "rxcui", "name", "n_openfda_labels"]
        )
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d candidates to %s", len(rows), config.candidates_path)


if __name__ == "__main__":
    main()
