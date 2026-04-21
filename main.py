import copy
import csv
import json
import os
import re
import sys
import uuid
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv
import pdfplumber

load_dotenv()

# ── Push Notification ─────────────────────────────────────────────────────────

def send_notification(title: str, message: str) -> bool:
    """Send a push notification via Pushover. Returns True on success."""
    user  = os.getenv("PUSHOVER_USER")
    token = os.getenv("PUSHOVER_TOKEN")

    if not user or not token:
        print("Pushover credentials not set in .env — skipping notification.", file=sys.stderr)
        return False

    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={"token": token, "user": user, "title": title, "message": message},
        timeout=10,
    )

    if response.status_code == 200:
        print(f"Notification sent: {title}", file=sys.stderr)
        return True

    print(f"Notification failed: {response.status_code} {response.text}", file=sys.stderr)
    return False


# ── PII Masking ────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

def _load_pii_rules(config_path: str = "pii_config.json") -> list:
    """Load enabled PII rules from config and compile their regex patterns."""
    resolved = Path(config_path) if Path(config_path).is_absolute() else _HERE / config_path
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if not config.get("enabled", True):
        return []
    rules = []
    for rule in config.get("rules", []):
        if rule.get("enabled", True):
            rules.append({
                "name": rule["name"],
                "token": rule["token"],
                "regex": re.compile(rule["pattern"]),
            })
    return rules


def _mask_string(value: str, rules: list) -> str:
    for rule in rules:
        value = rule["regex"].sub(rule["token"], value)
    return value


def _mask_node(node, rules: list):
    if isinstance(node, dict):
        return {k: _mask_node(v, rules) for k, v in node.items()}
    if isinstance(node, list):
        return [_mask_node(item, rules) for item in node]
    if isinstance(node, str):
        return _mask_string(node, rules)
    return node


def mask_pii(data: dict, config_path: str = "pii_config.json") -> dict:
    """Return a deep copy of data with all PII values replaced by tokens."""
    rules = _load_pii_rules(config_path)
    if not rules:
        return copy.deepcopy(data)
    return _mask_node(data, rules)


# ── Customer Lookup ───────────────────────────────────────────────────────────

def _extract_fields_from_json(json_data: dict) -> dict:
    """Pull all field values from every page into a flat dict with lowercase keys."""
    fields = {}
    for page in json_data.get("pages", []):
        # fields section
        for k, v in page.get("fields", {}).items():
            fields[k.strip().lower()] = v
        # table section — flatten all cell values using lowercase column headers
        for table in page.get("tables", []):
            for row in table:
                for k, v in row.items():
                    if k and v and v != "None":
                        fields[k.strip().lower()] = v
    return fields


def _normalise(value: str) -> str:
    """Lowercase and strip whitespace for loose comparison."""
    return value.strip().lower()


def customer_exists(json_path: str, csv_path: str = "") -> bool:
    """
    Read name, email and phone from the extracted JSON file and check
    whether any one of them matches a record in the CSV.
    Returns True if a match is found, False otherwise.
    """
    if not csv_path:
        csv_path = str(_HERE / "customer_data" / "customers.csv")
    json_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    fields = _extract_fields_from_json(json_data)

    # --- extract name, email, phone from JSON fields ---
    raw_name  = fields.get("name", "")
    raw_email = fields.get("email", "")
    raw_phone = fields.get("phone", "")

    # Clean mixed values (e.g. "Arjun Patel Make / Model: Toyota" → "Arjun Patel")
    name  = _normalise(raw_name.split("Make")[0].split("/")[0])
    email = _normalise(raw_email.split(" ")[0])                  # strip trailing junk
    phone = _normalise(re.split(r"\s+[A-Za-z]", raw_phone)[0])  # strip trailing text

    print(f"  Extracted  name='{name}'  email='{email}'  phone='{phone}'")

    if not any([name, email, phone]):
        print("  No name/email/phone found in JSON.")
        return False

    # --- compare against every row in the CSV (auto-detect delimiter) ---
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        f.seek(0)
        for row in csv.DictReader(f, dialect=dialect):
            csv_name  = _normalise(row.get("customer_name", ""))
            csv_email = _normalise(row.get("customer_email", ""))
            csv_phone = _normalise(row.get("customer_phone", ""))

            # name match: compare word sets so "Last, First" == "First Last"
            name_match = bool(name and csv_name and
                set(re.sub(r"[^a-z\s]", "", name).split()) ==
                set(re.sub(r"[^a-z\s]", "", csv_name).split()))

            if (
                name_match or
                (email and email == csv_email) or
                (phone and phone in csv_phone)
            ):
                print(f"  Match found  customer_id={row['customer_id']}  name='{row['customer_name']}'")
                return True

    print("  No match found in customer CSV.")
    return False


# ── Novated Lease ─────────────────────────────────────────────────────────────

def novated_lease(json_path: str) -> dict:
    """Calculate novated lease figures from net income in the extracted JSON."""
    json_data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    # Search all table cell values for one containing "NET INCOME"
    net_income = 0.0
    for page in json_data.get("pages", []):
        for table in page.get("tables", []):
            for row in table:
                for v in row.values():
                    if v and "net income" in str(v).lower():
                        m = re.search(r"[\d,]+\.\d+", str(v))
                        if m:
                            net_income = float(m.group().replace(",", ""))

    new_salary_after_lease = net_income - 500
    tax_savings = 500 * 12

    return {
        "net_income": net_income,
        "new_salary_after_lease": new_salary_after_lease,
        "tax_savings": tax_savings,
    }


# ── PDF Extraction ─────────────────────────────────────────────────────────────

def pdf_to_json(pdf_path: str) -> dict:
    result = {"pages": []}

    with pdfplumber.open(pdf_path) as pdf:
        result["total_pages"] = len(pdf.pages)

        for i, page in enumerate(pdf.pages, start=1):
            page_data = {"page": i, "fields": {}, "tables": [], "text": []}

            # --- tables ---
            for table in page.extract_tables():
                if not table:
                    continue
                headers = [str(c).strip() for c in table[0]]
                rows = []
                for row in table[1:]:
                    rows.append({headers[j]: str(c).strip() for j, c in enumerate(row) if j < len(headers)})
                page_data["tables"].append(rows)

            # --- text lines ---
            raw = page.extract_text() or ""
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = re.sub(r"\s+", "_", key.strip().lower())
                    key = re.sub(r"[^a-z0-9_]", "", key)
                    if key:
                        page_data["fields"][key] = val.strip()
                else:
                    page_data["text"].append(line)

            result["pages"].append(page_data)

    return result


# ── Car Claim CSV ──────────────────────────────────────────────────────────────

CLAIM_CSV = _HERE / "customer_data" / "car_claim.csv"
CLAIM_COLUMNS = [
    "claim_id", "customer_name", "customer_id",
    "summary_of_claim", "created_date", "status",
    "claim_details", "claim_document",
]


def _get_customer_id(customer_name: str, csv_path: str = "") -> str:
    """Look up customer_id from customers.csv by name."""
    if not csv_path:
        csv_path = str(_HERE / "customer_data" / "customers.csv")
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            sample = f.read(2048)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            f.seek(0)
            for row in csv.DictReader(f, dialect=dialect):
                csv_name = row.get("customer_name", "").strip().lower()
                if set(re.sub(r"[^a-z\s]", "", customer_name.lower()).split()) == \
                   set(re.sub(r"[^a-z\s]", "", csv_name).split()):
                    return row.get("customer_id", "")
    except FileNotFoundError:
        pass
    return ""


def add_car_claim(json_path: str, masked_json: dict, pdf_path: str,
                  status: str = "Open") -> str:
    """
    Create car_claim.csv if it does not exist, then append a new claim row.
    Returns the generated claim_id.
    """
    json_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    fields = _extract_fields_from_json(json_data)

    # --- derive fields ---
    raw_name = fields.get("name", "")
    customer_name = raw_name.split("Make")[0].split("/")[0].strip()

    customer_id = _get_customer_id(customer_name)
    summary = fields.get("description", fields.get("damage", ""))
    claim_id = "CLM-" + str(uuid.uuid4())[:8].upper()
    created_date = date.today().isoformat()
    claim_details = json.dumps(masked_json, ensure_ascii=False)

    row = {
        "claim_id":        claim_id,
        "customer_name":   customer_name,
        "customer_id":     customer_id,
        "summary_of_claim": summary,
        "created_date":    created_date,
        "status":          status,
        "claim_details":   claim_details,
        "claim_document":  str(Path(pdf_path).resolve()),
    }

    # create file with header if it doesn't exist
    file_exists = CLAIM_CSV.exists()
    with open(CLAIM_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CLAIM_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Claim recorded  claim_id={claim_id}", file=sys.stderr)
    return claim_id


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <file.pdf>")
        sys.exit(1)

    pdf_file = Path(sys.argv[1])
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # Raw extraction
    data = pdf_to_json(str(pdf_file))
    raw_file = out_dir / (pdf_file.stem + ".json")
    raw_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved raw     → {raw_file}", file=sys.stderr)

    # PII-masked
    masked = mask_pii(data)
    masked_file = out_dir / (pdf_file.stem + "_masked.json")
    masked_output = json.dumps(masked, indent=2, ensure_ascii=False)
    masked_file.write_text(masked_output, encoding="utf-8")
    print(f"Saved masked  → {masked_file}", file=sys.stderr)

    # Customer lookup + notification
    raw_json_path = str(raw_file)
    matched = customer_exists(raw_json_path)
    if matched:
        send_notification(
            title="Customer Match Found",
            message=f"Document '{pdf_file.name}' matched an existing customer in the CSV.",
        )
    else:
        send_notification(
            title="No Customer Match",
            message=f"Document '{pdf_file.name}' did not match any customer in the CSV.",
        )

    # Add claim record
    add_car_claim(raw_json_path, masked, str(pdf_file))

    print(masked_output)
