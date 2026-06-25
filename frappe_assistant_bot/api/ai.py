import frappe
import requests
import json
import re
import time

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_URL           = "http://localhost:11434/api/generate"
OLLAMA_MODEL         = "qwen3:8b"
TIMEOUT              = 120   # seconds — LLM responses can be slow
CONFIDENCE_THRESHOLD = 0.40  # below this, ask user to clarify instead of acting

# ── Supported actions ─────────────────────────────────────────────────────────
SUPPORTED_ACTIONS = {
    "create_document",
    "update_document",
    "delete_document",
    "get_document",
    "list_documents",
    "search_documents",
    "aggregate_documents",
    "group_documents",
}

# Fields that are tried (in order) when no exact name match exists.
# We exclude "description" here to avoid expensive LIKE queries on large fields.
_COMMON_DISPLAY_FIELDS = [
    "name",
    "title",
    "item_name",
    "customer_name",
    "supplier_name",
    "employee_name",
    "warehouse_name",
    "subject",
    "full_name",
    "account_name",
    "project_name",
    "lead_name",
]

# ── Base system instruction (unified single-pass prompt) ──────────────────────
_BASE_SYSTEM_INSTRUCTION = """You are a highly intelligent, ChatGPT-level ERPNext/Frappe AI assistant.
Your job is to read the user's message (which may contain spelling errors, typos, abbreviations, broken English, or indirect business references) and map it to the correct ERPNext action in a single JSON structure.

Return ONLY a valid JSON object. No explanation, no markdown formatting, no code fences. Just the raw JSON.

JSON Response Format:
{
  "normalized_prompt": "<clean, corrected English restatement of the user's intent>",
  "confidence": <float between 0.0 and 1.0 representing your understanding confidence>,
  "suggestions": ["Alternative phrasing 1", "Alternative phrasing 2"],
  "action": "<one of: create_document, update_document, delete_document, get_document, list_documents, search_documents, aggregate_documents, group_documents>",
  "doctype": "<detected ERPNext DocType name>",
  "name": "<resolved document name/ID or a identifier like 'Metamorphosis' or 'Franz Kafka'>",
  "data": { <key-value pairs for create_document or update_document> },
  "filters": { <key-value filters for search_documents or aggregate_documents> },
  "operation": "<count for aggregate_documents>",
  "group_by": "<fieldname for group_documents>",
  "limit": 20
}

CRITICAL INSTRUCTIONS

You are an ERP ACTION PLANNER.

You are NOT allowed to answer the user's question.

You are NOT allowed to provide records.

You are NOT allowed to provide search results.

You are NOT allowed to provide explanations.

You are NOT allowed to generate example data.

You are NOT allowed to generate business data.

You are NOT allowed to return a list of customers, books, items, invoices, suppliers, warehouses, or any records.

You MUST ONLY generate an action JSON object.

The backend system will execute the action.

If the user asks:

"Show all customers"

You MUST return:

{
"action":"list_documents",
"doctype":"Customer",
"limit":20
}

If the user asks:

"Count books"

You MUST return:

{
"action":"aggregate_documents",
"doctype":"Book",
"operation":"count"
}

Returning actual records is forbidden.

Returning explanations is forbidden.

Returning natural language is forbidden.

Returning partial JSON is forbidden.

You must return exactly one JSON object.

INTENT MAPPING & REASONING RULES:
1. Spelling and Abbreviation Correction:
   - Fix obvious typos: "avilable/issue/custmrs/itm/wharehouse" -> "Available/Issued/Customer/Item/Warehouse".
   - Expand abbreviations: "hw many" -> "how many", "updte" -> "update", "delet" -> "delete", "creat" -> "create".
   - Example: "show avilable books" -> action: "search_documents", doctype: "Book", filters: {"status": "Available"}

2. Search and Filtering:
   - Match attributes to correct metadata fieldnames. If user says "show kafkas books", look at Book metadata to see "author" field, and use filters: {"author": "Kafka"} or {"author": "Franz Kafka"}.
   - For time-based filters (e.g. "books added today", "invoices from yesterday"), use standard field "creation" (format: "YYYY-MM-DD" or date ranges if known). Note: standard fields like "creation" (datetime) are always available.
   - Example: "show invoices of abc" -> action: "search_documents", doctype: "Sales Invoice", filters: {"customer_name": "abc"} (or other field matching supplier/customer in the metadata).
   - Example: "show supplier from pune" -> action: "search_documents", doctype: "Supplier", filters: {"city": "Pune"} (or other field matching location).

3. Document Creation & Updates:
   - "create customer abc pvt ltd" -> action: "create_document", doctype: "Customer", data: {"customer_name": "abc pvt ltd"}
   - "updte book status issued" -> action: "update_document", doctype: "Book", name: "book", data: {"status": "Issued"}. Since name is unspecified, set "name": "book" and the system will try to resolve it.
   - "delete kafka book" -> action: "delete_document", doctype: "Book", name: "kafka".

4. Analytics and Counting:
   - "how many books are available" -> action: "aggregate_documents", doctype: "Book", operation: "count", filters: {"status": "Available"}
   - "what stock do we have" -> action: "list_documents", doctype: "Item"
   - "how many books exist" or "count books" -> action: "aggregate_documents", doctype: "Book", operation: "count"
   - "which books are issued" -> action: "search_documents", doctype: "Book", filters: {"status": "Issued"}

5. Field and Option Constraints:
   - Use ONLY the fieldnames and Select options provided in the DocType metadata below.
   - Do NOT invent field names or option values.
"""

_NORMALIZATION_INSTRUCTION = """\
You are an intent normalization engine for an ERPNext assistant.
Return ONLY valid JSON.
"""

def _normalize_prompt(prompt_clean):
    # Legacy wrapper to avoid breaking any direct imports/calls.
    # Returns normalized prompt and confidence.
    return {
        "normalized": prompt_clean,
        "confidence": 1.0,
        "suggestions": [],
    }

# ── Performance Profiling Helper ──────────────────────────────────────────────
def _log_perf(t_total, t_norm_gen, t_metadata, t_resolve, t_db, t_format, t_validation):
    msg = (
        "[AI PERF]\n"
        "Total: {:.3f}s\n\n"
        "Normalization & Action Gen: {:.3f}s\n"
        "Metadata Loading: {:.3f}s\n"
        "Entity Resolution: {:.3f}s\n"
        "Database Query: {:.3f}s\n"
        "Response Formatting: {:.3f}s\n"
        "Validation: {:.3f}s".format(
            t_total, t_norm_gen, t_metadata, t_resolve, t_db, t_format, t_validation
        )
    )
    # Log as warning so it is always captured in frappe.log regardless of log level
    frappe.logger().warning(msg)
    # Also print to stdout for CLI testing convenience
    print(msg)

# ── Heuristic & Fast Path Layers ──────────────────────────────────────────────
def _detect_doctype_heuristically(prompt, available_doctypes):
    """
    Heuristically detect target DocType from prompt words in <1ms.
    Matches plurals and common abbreviations/typos.
    """
    p = prompt.lower()
    words = re.findall(r'[a-z]+', p)
    
    mappings = {
        "buk": "Book", "buks": "Book", "book": "Book", "books": "Book",
        "cust": "Customer", "custmer": "Customer", "custmers": "Customer", "customer": "Customer", "customers": "Customer",
        "itm": "Item", "itms": "Item", "item": "Item", "items": "Item",
        "wharehouse": "Warehouse", "warhouse": "Warehouse", "warehouse": "Warehouse", "warehouses": "Warehouse",
        "suplr": "Supplier", "supplier": "Supplier", "suppliers": "Supplier",
        "emplyee": "Employee", "employee": "Employee", "employees": "Employee",
        "lead": "Lead", "leads": "Lead",
        "task": "Task", "tasks": "Task"
    }
    for w in words:
        if w in mappings:
            return mappings[w]
            
    import difflib
    doctype_lower_map = {d.lower(): d for d in available_doctypes}
    
    for w in words:
        if w in doctype_lower_map:
            return doctype_lower_map[w]
        if w.endswith('s') and w[:-1] in doctype_lower_map:
            return doctype_lower_map[w[:-1]]
        if w.endswith('es') and w[:-2] in doctype_lower_map:
            return doctype_lower_map[w[:-2]]
            
    for w in words:
        if len(w) > 3:
            matches = difflib.get_close_matches(w, list(doctype_lower_map.keys()), n=1, cutoff=0.7)
            if matches:
                return doctype_lower_map[matches[0]]
                
    return None

def _check_fast_path(prompt):
    """
    Bypass Ollama completely for extremely obvious list/count queries with no filters or conditions.
    Used only when confidence is extremely high.
    """
    p = prompt.strip().lower().rstrip('?.!')
    p = re.sub(r'\s+', ' ', p)

    # 1. Simple List / Show all patterns with NO filters or status
    # e.g., "show all customer", "list all item", "show books", "list warehouses"
    list_match = re.match(
        r'^(show all|list all|list|show)\s+(books|book|customers|customer|warehouses|warehouse|items|item|suppliers|supplier|employees|employee|leads|lead|tasks|task)$',
        p
    )
    if list_match:
        doctype_word = list_match.group(2)
        doctype_map = {
            "book": "Book", "books": "Book",
            "customer": "Customer", "customers": "Customer",
            "warehouse": "Warehouse", "warehouses": "Warehouse",
            "item": "Item", "items": "Item",
            "supplier": "Supplier", "suppliers": "Supplier",
            "employee": "Employee", "employees": "Employee",
            "lead": "Lead", "leads": "Lead",
            "task": "Task", "tasks": "Task"
        }
        dt = doctype_map.get(doctype_word)
        if dt:
            return {
                "success": True,
                "action": "list_documents",
                "doctype": dt,
                "limit": 20,
                "confidence": 1.0,
                "normalized_prompt": "List all {} records".format(dt)
            }

    # 2. Simple Count patterns with NO filters
    # e.g., "count customer", "count books", "how many items exist"
    count_patterns = [
        r'^(count)\s+(books|book|customers|customer|items|item|suppliers|supplier|employees|employee|leads|lead|tasks|task)$',
        r'^how many\s+(books|book|customers|customer|items|item|suppliers|supplier|employees|employee|leads|lead|tasks|task)\s+(exist|do we have|are there)$',
        r'^how many\s+(books|book|customers|customer|items|item|suppliers|supplier|employees|employee|leads|lead|tasks|task)$'
    ]
    for pattern in count_patterns:
        count_match = re.match(pattern, p)
        if count_match:
            groups = count_match.groups()
            doctype_word = None
            for g in groups:
                if g in ("books", "book", "customers", "customer", "items", "item", "suppliers", "supplier", "employees", "employee", "leads", "lead", "tasks", "task"):
                    doctype_word = g
                    break
            doctype_map = {
                "book": "Book", "books": "Book",
                "customer": "Customer", "customers": "Customer",
                "item": "Item", "items": "Item",
                "supplier": "Supplier", "suppliers": "Supplier",
                "employee": "Employee", "employees": "Employee",
                "lead": "Lead", "leads": "Lead",
                "task": "Task", "tasks": "Task"
            }
            if doctype_word:
                dt = doctype_map.get(doctype_word)
                if dt:
                    return {
                        "success": True,
                        "action": "aggregate_documents",
                        "doctype": dt,
                        "operation": "count",
                        "filters": {},
                        "confidence": 1.0,
                        "normalized_prompt": "Count all {} records".format(dt)
                    }

    return None


def _get_candidate_doctypes(prompt, available_doctypes):
    """
    Identify candidate DocTypes from the user's prompt.
    Uses scoring to return the top 3 best matching DocTypes deterministically.
    """
    p = prompt.lower()
    words = re.findall(r'[a-z0-9]+', p)

    aliases = {
        "stock": ["Item", "Warehouse", "Bin"],
        "stocks": ["Item", "Warehouse", "Bin"],
        "invoice": ["Sales Invoice", "Purchase Invoice"],
        "invoices": ["Sales Invoice", "Purchase Invoice"],
        "billing": ["Sales Invoice", "Purchase Invoice"],
        "transaction": ["Journal Entry", "Payment Entry", "Sales Invoice"],
        "transactions": ["Journal Entry", "Payment Entry", "Sales Invoice"],
        "product": ["Item"],
        "products": ["Item"],
        "vendor": ["Supplier"],
        "vendors": ["Supplier"],
        "staff": ["Employee"],
        "todo": ["ToDo"],
        "todos": ["ToDo"],
    }

    candidate_scores = {}

    def add_score(c, score):
        if c in available_doctypes:
            candidate_scores[c] = max(candidate_scores.get(c, 0), score)

    for w in words:
        if w in aliases:
            for dt in aliases[w]:
                add_score(dt, 50)

    doctype_lower_map = {d.lower(): d for d in available_doctypes}
    for w in words:
        if w in doctype_lower_map:
            dt = doctype_lower_map[w]
            add_score(dt, 80 + len(dt))
        if w.endswith('s') and w[:-1] in doctype_lower_map:
            dt = doctype_lower_map[w[:-1]]
            add_score(dt, 80 + len(dt))
        if w.endswith('es') and w[:-2] in doctype_lower_map:
            dt = doctype_lower_map[w[:-2]]
            add_score(dt, 80 + len(dt))

    import difflib
    for dt in available_doctypes:
        dt_clean = dt.lower()
        if dt_clean in p:
            add_score(dt, 100 + len(dt))
            continue
        
        dt_words = dt_clean.split()
        dt_words_len = len(dt_words)
        prompt_words = p.split()
        if len(prompt_words) >= dt_words_len:
            for i in range(len(prompt_words) - dt_words_len + 1):
                sub = " ".join(prompt_words[i:i+dt_words_len])
                ratio = difflib.SequenceMatcher(None, dt_clean, sub).ratio()
                if ratio > 0.85:
                    add_score(dt, int(ratio * 90) + len(dt))
                    break

    sorted_candidates = sorted(candidate_scores.keys(), key=lambda x: candidate_scores[x], reverse=True)
    return sorted_candidates[:3]

# ── DocType Metadata Helpers ──────────────────────────────────────────────────

def get_available_doctypes():
    """
    Return a list of all non-custom DocType names in the system (cached).
    """
    cache_key = "ai_assistant:available_doctypes"
    cached = frappe.cache().get_value(cache_key)
    if cached:
        return cached
        
    records = frappe.get_all(
        "DocType",
        filters={"custom": 0},
        fields=["name"],
    )
    names = [r["name"] for r in records]
    frappe.cache().set_value(cache_key, names, expires_in_sec=3600)
    return names


def get_doctype_metadata(doctype):
    """
    Return relevant field descriptors for *doctype*.
    Only sends required, display, filter, select, or status-related fields to keep prompt small.
    Caches results for 1 hour.
    """
    cache_key = "ai_assistant:metadata:{}".format(doctype)
    cached = frappe.cache().get_value(cache_key)
    if cached:
        return cached

    SKIP_TYPES = {
        "Section Break", "Column Break", "HTML", "Tab Break",
        "Fold", "Heading", "Button", "Image", "Table", "Table MultiSelect"
    }
    try:
        meta = frappe.get_meta(doctype)
    except Exception:
        return []

    fields = []
    EXCLUDED_FIELDNAMES = {
        "chart_options", "custom_options", "filters_json", "dynamic_filters_json",
        "color", "background_color"
    }
    for f in meta.fields:
        if f.fieldname in EXCLUDED_FIELDNAMES:
            continue
        if f.fieldtype in SKIP_TYPES:
            continue
        if f.hidden:
            continue
        if not f.fieldname:
            continue
        
        fields.append({
            "fieldname": f.fieldname,
            "label":     f.label or f.fieldname,
            "fieldtype": f.fieldtype,
            "reqd":      bool(f.reqd),
            "options":   f.options or "",
        })

    # Cache for 1 hour
    frappe.cache().set_value(cache_key, fields, expires_in_sec=3600)
    return fields


def _build_metadata_block(doctype, fields):
    """
    Render a human-readable metadata block to inject into the system prompt.
    """
    if not fields:
        return ""

    lines = [
        "\n--- DocType Metadata ---",
        "DocType: {}".format(doctype),
        "Available fields (use ONLY these fieldnames in 'data' or 'filters'):",
    ]
    for f in fields:
        req_marker = " [REQUIRED]" if f["reqd"] else ""
        if f["fieldtype"] == "Select" and f["options"]:
            # List each option on its own sub-line so the AI can pick valid values
            raw_options = [o.strip() for o in f["options"].split("\n") if o.strip()]
            if len(raw_options) > 15:
                truncated_options = raw_options[:10]
                omitted_count = len(raw_options) - 10
                option_lines = "\n".join("      - {}".format(o) for o in truncated_options)
                option_lines += "\n      - ... and {} more options omitted".format(omitted_count)
            else:
                option_lines = "\n".join("      - {}".format(o) for o in raw_options)
            lines.append(
                "  - {} (Select{})".format(f["fieldname"], req_marker)
            )
            lines.append("    Options:\n" + option_lines)
        else:
            if len(f["options"]) > 150:
                opts = " | options: (omitted due to length)"
            else:
                opts = " | options: {}".format(f["options"]) if f["options"] else ""
            lines.append(
                "  - {} ({}{}{})".format(
                    f["fieldname"], f["fieldtype"], req_marker, opts
                )
            )
    lines.append("--- End Metadata ---\n")
    return "\n".join(lines)


def _build_system_prompt(candidates=None):
    """
    Compose the final system prompt, injecting metadata blocks for all candidate DocTypes.
    Also injects the current local date to allow precise time-based filtering.
    """
    prompt = _BASE_SYSTEM_INSTRUCTION
    
    # Inject current date to help with date filtering (e.g. today/yesterday)
    current_date = frappe.utils.today() if hasattr(frappe, "utils") else time.strftime("%Y-%m-%d")
    prompt += "\n\nCURRENT CONTEXT:\n- Current Date: {}\n".format(current_date)
    
    if candidates:
        for dt in candidates:
            fields = get_doctype_metadata(dt)
            prompt += _build_metadata_block(dt, fields)
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Metadata-Aware Field Validators
# ─────────────────────────────────────────────────────────────────────────────

def _validate_fields_against_metadata(data, fields):
    """
    Cross-check the AI-supplied *data* dict against *fields* (from
    get_doctype_metadata).  Returns a list of error strings (empty = OK).

    Rules:
    1. Reject any key in *data* that is not a known fieldname.
    2. Warn about missing required fields.
    """
    if not fields:
        # No metadata available — skip validation gracefully
        return []

    known      = {f["fieldname"] for f in fields}
    required   = {f["fieldname"] for f in fields if f["reqd"]}
    errors     = []

    # 1. Unknown fields
    for key in data.keys():
        if key not in known:
            errors.append("Unknown field '{}' — not present in DocType metadata.".format(key))

    # 2. Missing required fields
    for fieldname in required:
        if fieldname not in data:
            errors.append("Missing required field: {}".format(fieldname))

    return errors


def _get_searchable_fields(doctype, fields_meta):
    """
    Build the set of fields to search during entity resolution.
    Always includes the common display fields that exist on *doctype*.
    Adds all Data, Link, or Select fields from metadata to ensure accuracy.
    """
    all_fieldnames = {f["fieldname"] for f in fields_meta} if fields_meta else set()

    candidates = set()
    for fn in _COMMON_DISPLAY_FIELDS:
        if fn == "name" or fn in all_fieldnames:
            candidates.add(fn)

    # Include all Data, Link, and Select fields to allow search by fields like author, etc.
    for f in (fields_meta or []):
        if f["fieldtype"] in ("Data", "Link", "Select"):
            candidates.add(f["fieldname"])

    return list(candidates)


def _build_clarification_response(doctype, query, records, search_fields):
    matches = []
    for r in records:
        label_parts = [r["name"]]
        for fn in search_fields:
            if fn == "name":
                continue
            val = r.get(fn)
            if val and val != r["name"]:
                label_parts.append("{}: {}".format(fn, val))
                break  # Keep the label clean by displaying only one main attribute
        matches.append({
            "name":  r["name"],
            "label": " | ".join(label_parts),
        })
    return {
        "needs_clarification": True,
        "doctype":  doctype,
        "query":    query,
        "matches":  matches,
        "message":  (
            "Found {} {}s matching '{}'. "
            "Which one would you like to use?".format(
                len(records), doctype, query
            )
        ),
    }


def resolve_document(doctype, search_value, fields_meta=None):
    """
    Resolve natural-language identifiers to document names.
    Implements a weighted fallback mechanism: exact -> startswith -> contains -> fuzzy.
    Instrumented with perf timing.
    """
    t_start = time.perf_counter()
    try:
        if not search_value:
            return {"error": "No document identifier provided."}

        search_val_str = str(search_value).strip()

        frappe.logger().info(
            "AI Assistant — resolving {} '{}'".format(doctype, search_val_str)
        )

        # 1. Check exact name match first
        if frappe.db.exists(doctype, search_val_str):
            frappe.logger().info(
                "AI Assistant — exact name match: {}".format(search_val_str)
            )
            return {"resolved_name": search_val_str, "confidence": 1.0}

        searchable = _get_searchable_fields(doctype, fields_meta or [])
        search_fields = [fn for fn in searchable if fn != "name"]

        if not search_fields:
            search_fields = ["name"]

        # Priority 1: Exact match on search fields
        records = []
        for fn in search_fields:
            if fn == "name":
                continue
            exact_records = frappe.get_all(
                doctype,
                filters={fn: search_val_str},
                fields=["name"] + [f for f in search_fields if f != "name"][:3]
            )
            if exact_records:
                records.extend(exact_records)

        # Remove duplicates
        seen = set()
        unique_records = []
        for r in records:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique_records.append(r)

        if len(unique_records) == 1:
            return {"resolved_name": unique_records[0]["name"], "confidence": 1.0}
        elif len(unique_records) > 1:
            return _build_clarification_response(doctype, search_val_str, unique_records, search_fields)

        # Priority 2: Startswith match on search fields
        records = []
        for fn in search_fields:
            if fn == "name":
                continue
            startswith_records = frappe.get_all(
                doctype,
                filters=[[doctype, fn, "like", "{}%".format(search_val_str)]],
                fields=["name"] + [f for f in search_fields if f != "name"][:3]
            )
            if startswith_records:
                records.extend(startswith_records)

        for r in records:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique_records.append(r)

        if len(unique_records) == 1:
            return {"resolved_name": unique_records[0]["name"], "confidence": 0.9}
        elif len(unique_records) > 1:
            return _build_clarification_response(doctype, search_val_str, unique_records, search_fields)

        # Priority 3: Contains match on search fields
        records = []
        for fn in search_fields:
            if fn == "name":
                continue
            contains_records = frappe.get_all(
                doctype,
                filters=[[doctype, fn, "like", "%{}%".format(search_val_str)]],
                fields=["name"] + [f for f in search_fields if f != "name"][:3]
            )
            if contains_records:
                records.extend(contains_records)

        for r in records:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique_records.append(r)

        if len(unique_records) == 1:
            return {"resolved_name": unique_records[0]["name"], "confidence": 0.8}
        elif len(unique_records) > 1:
            return _build_clarification_response(doctype, search_val_str, unique_records, search_fields)

        # Priority 4: Fuzzy similarity match
        all_records = frappe.get_all(
            doctype,
            fields=["name"] + [f for f in search_fields if f != "name"][:3],
            limit_page_length=100
        )

        import difflib
        fuzzy_matches = []
        for r in all_records:
            best_score = 0.0
            best_score = max(best_score, difflib.SequenceMatcher(None, search_val_str.lower(), r["name"].lower()).ratio())
            for fn in search_fields:
                if fn == "name":
                    continue
                val = r.get(fn)
                if val:
                    score = difflib.SequenceMatcher(None, search_val_str.lower(), str(val).lower()).ratio()
                    if score > best_score:
                        best_score = score
            if best_score >= 0.6:
                fuzzy_matches.append((best_score, r))

        fuzzy_matches.sort(key=lambda x: x[0], reverse=True)

        if fuzzy_matches:
            if len(fuzzy_matches) == 1:
                return {"resolved_name": fuzzy_matches[0][1]["name"], "confidence": fuzzy_matches[0][0]}
            elif fuzzy_matches[0][0] >= 0.85 and (fuzzy_matches[0][0] - fuzzy_matches[1][0]) >= 0.15:
                return {"resolved_name": fuzzy_matches[0][1]["name"], "confidence": fuzzy_matches[0][0]}
            else:
                candidates = [item[1] for item in fuzzy_matches[:5]]
                return _build_clarification_response(doctype, search_val_str, candidates, search_fields)

        return {"error": "No matching {} found for '{}'.".format(doctype, search_val_str)}

    finally:
        t_elapsed = time.perf_counter() - t_start
        if getattr(frappe.flags, "perf_times", None):
            frappe.flags.perf_times["resolve"] += t_elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Parsing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_think_tags(text):
    """Remove <think>…</think> blocks that Qwen3 emits in reasoning mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text):
    """
    Best-effort JSON extractor.
    Enforces maximum 10KB response size, trims trailing characters, removes duplicate keys,
    logs malformed responses, and enforces allowed top-level keys schema.
    """
    if not isinstance(text, str):
        raise ValueError("Model output is not a string.")

    # 1. Reject responses larger than 10KB
    if len(text) > 10240:
        frappe.logger().error("Model response exceeded 10KB ({} bytes)".format(len(text)))
        raise ValueError("Model response too large (exceeds 10KB).")

    # 2. Trim output after final closing brace
    last_brace_idx = text.rfind('}')
    if last_brace_idx != -1:
        text = text[:last_brace_idx + 1]

    def _parse_pairs(pairs):
        # Convert pairs to dict to keep the last duplicate key value
        return dict(pairs)

    parsed = None

    # Attempt 1 — direct parse
    try:
        parsed = json.loads(text, object_pairs_hook=_parse_pairs)
    except json.JSONDecodeError:
        pass

    if parsed is None:
        # Attempt 2 — strip code fences
        fence_stripped = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
        try:
            parsed = json.loads(fence_stripped, object_pairs_hook=_parse_pairs)
        except json.JSONDecodeError:
            pass

    if parsed is None:
        # Attempt 3 — extract first {...} block
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(), object_pairs_hook=_parse_pairs)
            except json.JSONDecodeError:
                pass

    if parsed is None:
        frappe.logger().error("Malformed LLM response: {}".format(text))
        raise ValueError("No valid JSON found in model output.")

    # Enforce allowed top-level keys schema
    if isinstance(parsed, dict):
        allowed_keys = {
            "confidence", "normalized_prompt", "tool", "action", "doctype", "name",
            "data", "filters", "operation", "group_by", "report_name",
            "workflow_action", "limit", "suggestions"
        }
        parsed = {k: v for k, v in parsed.items() if k in allowed_keys}

    return parsed


def _validate_action(parsed):
    """
    Validate the parsed JSON has the required fields for its action.
    Raises ValueError with a descriptive message on failure.
    """
    action = parsed.get("action")
    if not action:
        raise ValueError("Missing field: action")

    if action not in SUPPORTED_ACTIONS:
        raise ValueError(
            "Unsupported action: {}. Supported: {}.".format(
                action, ", ".join(sorted(SUPPORTED_ACTIONS))
            )
        )

    doctype = parsed.get("doctype")
    if not doctype:
        raise ValueError("Missing field: doctype")

    if action == "create_document":
        if not isinstance(parsed.get("data"), dict):
            raise ValueError("'create_document' requires a 'data' object.")

    elif action == "update_document":
        if not parsed.get("name"):
            raise ValueError("Missing field: name")
        if not isinstance(parsed.get("data"), dict):
            raise ValueError("'update_document' requires a 'data' object.")

    elif action == "delete_document":
        if not parsed.get("name"):
            raise ValueError("Missing field: name")

    elif action == "get_document":
        if not parsed.get("name"):
            raise ValueError("Missing field: name")

    elif action == "list_documents":
        pass

    elif action == "search_documents":
        pass

    elif action == "aggregate_documents":
        if not parsed.get("operation"):
            raise ValueError("Missing field: operation")

    elif action == "group_documents":
        if not parsed.get("group_by"):
            raise ValueError("Missing field: group_by")


# ─────────────────────────────────────────────────────────────────────────────
# Action Handlers
# ─────────────────────────────────────────────────────────────────────────────

def handle_create_document(parsed):
    """
    Insert a new Frappe document.

    Expected keys: doctype, data
    """
    doctype = parsed["doctype"]
    data    = parsed["data"]

    frappe.logger().info(
        "AI Assistant — CREATE {} | data: {}".format(doctype, json.dumps(data))
    )

    try:
        doc = frappe.get_doc({"doctype": doctype, **data})
        doc.insert(ignore_permissions=True)
        frappe.db.commit()

        msg = "{} {} created successfully".format(doc.doctype, doc.name)
        frappe.logger().info("AI Assistant — " + msg)
        return {
            "success":          True,
            "action":           "create_document",
            "document_created": True,
            "doctype":          doc.doctype,
            "name":             doc.name,
            "message":          msg,
        }

    except frappe.MandatoryError as e:
        return {"success": False, "error": "Mandatory field missing: {}".format(str(e))}
    except frappe.ValidationError as e:
        return {"success": False, "error": "Validation error: {}".format(str(e))}
    except frappe.DuplicateEntryError as e:
        return {"success": False, "error": "Duplicate entry: {}".format(str(e))}
    except frappe.DoesNotExistError:
        return {"success": False, "error": "DocType '{}' does not exist.".format(doctype)}
    except Exception as e:
        frappe.log_error(
            frappe.get_traceback(),
            "AI Assistant — CREATE error [{} | {}]".format(doctype, json.dumps(data))
        )
        return {"success": False, "error": "Failed to create {}: {}".format(doctype, str(e))}


def handle_update_document(parsed, fields_meta=None):
    """
    Update fields on an existing Frappe document.

    Expected keys: doctype, name, data
    Supports natural-language names via resolve_document().
    """
    doctype = parsed["doctype"]
    name    = parsed["name"]
    data    = parsed["data"]

    # ── Phase 6: Entity Resolution ──────────────────────────────────────────
    resolution = resolve_document(doctype, name, fields_meta)
    if "needs_clarification" in resolution:
        return {
            "success":            False,
            "needs_clarification": True,
            "action":             "update_document",
            "doctype":            doctype,
            "matches":            resolution["matches"],
            "message":            resolution["message"],
        }
    if "error" in resolution:
        return {"success": False, "error": resolution["error"]}
    resolved_name = resolution["resolved_name"]

    frappe.logger().info(
        "AI Assistant — UPDATE {} {} | data: {}".format(doctype, resolved_name, json.dumps(data))
    )

    try:
        doc = frappe.get_doc(doctype, resolved_name)
        for field, value in data.items():
            setattr(doc, field, value)
        doc.save(ignore_permissions=True)
        frappe.db.commit()

        msg = "{} {} updated successfully".format(doctype, resolved_name)
        frappe.logger().info("AI Assistant — " + msg)
        return {
            "success": True,
            "action":  "update_document",
            "doctype": doctype,
            "name":    resolved_name,
            "message": msg,
            "updated_fields": list(data.keys()),
        }

    except frappe.DoesNotExistError:
        return {"success": False, "error": "{} '{}' not found.".format(doctype, resolved_name)}
    except frappe.ValidationError as e:
        return {"success": False, "error": "Validation error: {}".format(str(e))}
    except Exception as e:
        frappe.log_error(
            frappe.get_traceback(),
            "AI Assistant — UPDATE error [{} {}]".format(doctype, resolved_name)
        )
        return {"success": False, "error": "Failed to update {}: {}".format(doctype, str(e))}


def handle_delete_document(parsed, fields_meta=None):
    """
    Delete a Frappe document.

    Expected keys: doctype, name
    Supports natural-language names via resolve_document().
    """
    doctype = parsed["doctype"]
    name    = parsed["name"]

    # ── Phase 6: Entity Resolution ──────────────────────────────────────────
    resolution = resolve_document(doctype, name, fields_meta)
    if "needs_clarification" in resolution:
        return {
            "success":            False,
            "needs_clarification": True,
            "action":             "delete_document",
            "doctype":            doctype,
            "matches":            resolution["matches"],
            "message":            resolution["message"],
        }
    if "error" in resolution:
        return {"success": False, "error": resolution["error"]}
    resolved_name = resolution["resolved_name"]

    frappe.logger().info(
        "AI Assistant — DELETE {} {}".format(doctype, resolved_name)
    )

    try:
        frappe.delete_doc(doctype, resolved_name, ignore_permissions=True)
        frappe.db.commit()

        msg = "{} {} deleted successfully".format(doctype, resolved_name)
        frappe.logger().info("AI Assistant — " + msg)
        return {
            "success": True,
            "action":  "delete_document",
            "doctype": doctype,
            "name":    resolved_name,
            "message": msg,
        }

    except frappe.DoesNotExistError:
        return {"success": False, "error": "{} '{}' not found.".format(doctype, resolved_name)}
    except frappe.LinkExistsError as e:
        return {
            "success": False,
            "error": (
                "Cannot delete {} '{}': it is referenced by other records. "
                "Details: {}".format(doctype, resolved_name, str(e))
            ),
        }
    except Exception as e:
        frappe.log_error(
            frappe.get_traceback(),
            "AI Assistant — DELETE error [{} {}]".format(doctype, resolved_name)
        )
        return {"success": False, "error": "Failed to delete {}: {}".format(doctype, str(e))}


def handle_get_document(parsed, fields_meta=None):
    """
    Fetch a single Frappe document and return its fields as a dict.

    Expected keys: doctype, name
    Supports natural-language names via resolve_document().
    """
    doctype = parsed["doctype"]
    name    = parsed["name"]

    # ── Phase 6: Entity Resolution ──────────────────────────────────────────
    resolution = resolve_document(doctype, name, fields_meta)
    if "needs_clarification" in resolution:
        return {
            "success":            False,
            "needs_clarification": True,
            "action":             "get_document",
            "doctype":            doctype,
            "matches":            resolution["matches"],
            "message":            resolution["message"],
        }
    if "error" in resolution:
        return {"success": False, "error": resolution["error"]}
    resolved_name = resolution["resolved_name"]

    frappe.logger().info(
        "AI Assistant — GET {} {}".format(doctype, resolved_name)
    )

    try:
        doc = frappe.get_doc(doctype, resolved_name)
        doc_dict = doc.as_dict()

        # Remove internal Frappe meta fields for a cleaner response
        for key in ("docstatus", "idx", "owner", "creation", "modified",
                    "modified_by", "doctype", "__islocal", "__unsaved"):
            doc_dict.pop(key, None)

        frappe.logger().info(
            "AI Assistant — GET {} {} succeeded".format(doctype, resolved_name)
        )
        return {
            "success":  True,
            "action":   "get_document",
            "doctype":  doctype,
            "name":     resolved_name,
            "document": doc_dict,
            "message":  "Fetched {} {}".format(doctype, resolved_name),
        }

    except frappe.DoesNotExistError:
        return {"success": False, "error": "{} '{}' not found.".format(doctype, resolved_name)}
    except Exception as e:
        frappe.log_error(
            frappe.get_traceback(),
            "AI Assistant — GET error [{} {}]".format(doctype, resolved_name)
        )
        return {"success": False, "error": "Failed to fetch {}: {}".format(doctype, str(e))}


def handle_list_documents(parsed):
    """
    Return a list of documents of the given doctype.

    Expected keys: doctype, limit (optional, default 20)
    """
    doctype = parsed["doctype"]
    limit   = int(parsed.get("limit", 20))
    limit   = min(limit, 100)   # hard cap to avoid accidental large queries
    filters = parsed.get("filters", {})

    frappe.logger().info(
        "AI Assistant — LIST {} limit={} filters={}".format(
            doctype, limit, json.dumps(filters)
        )
    )

    try:
        records = frappe.get_all(
            doctype,
            filters=filters or {},
            fields=["name"],
            limit=limit,
        )

        frappe.logger().info(
            "AI Assistant — LIST {} returned {} records".format(doctype, len(records))
        )
        return {
            "success": True,
            "action":  "list_documents",
            "doctype": doctype,
            "count":   len(records),
            "records": [r["name"] for r in records],
            "message": "Found {} {} record(s)".format(len(records), doctype),
        }

    except frappe.DoesNotExistError:
        return {"success": False, "error": "DocType '{}' does not exist.".format(doctype)}
    except Exception as e:
        frappe.log_error(
            frappe.get_traceback(),
            "AI Assistant — LIST error [{}]".format(doctype)
        )
        return {"success": False, "error": "Failed to list {}: {}".format(doctype, str(e))}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Smart Filtering / Natural Language Search
# ─────────────────────────────────────────────────────────────────────────────

def _validate_search_filters(filters, fields_meta):
    """
    Validate filters supplied by the AI against live DocType metadata.

    Checks:
    1. Every filter key must be a known fieldname.
    2. For Select fields the value must be one of the listed options.

    Returns a list of error strings (empty = all good).
    """
    if not fields_meta or not filters:
        return []

    meta_map = {f["fieldname"]: f for f in fields_meta}
    errors   = []

    for key, value in filters.items():
        if key not in meta_map:
            errors.append("Unknown filter field '{}' — not present in DocType metadata.".format(key))
            continue
        field = meta_map[key]
        if field["fieldtype"] == "Select" and field["options"] and value:
            valid_opts = [o.strip() for o in field["options"].split("\n") if o.strip()]
            if str(value) not in valid_opts:
                errors.append(
                    "Invalid value '{}' for Select field '{}'. "
                    "Valid options: {}".format(value, key, ", ".join(valid_opts))
                )
    return errors


def handle_search_documents(parsed, fields_meta=None):
    """
    Search documents with AI-supplied filters and return full field data.

    Expected keys: doctype, filters (optional), limit (optional)
    """
    doctype = parsed["doctype"]
    filters = parsed.get("filters") or {}
    limit   = int(parsed.get("limit", 20))
    limit   = min(limit, 100)

    frappe.logger().info(
        "AI Assistant — SEARCH {} filters={} limit={}".format(
            doctype, json.dumps(filters), limit
        )
    )

    # Validate filters against metadata
    if fields_meta:
        filter_errors = _validate_search_filters(filters, fields_meta)
        if filter_errors:
            frappe.logger().warning(
                "AI Assistant — search filter errors: {}".format(filter_errors)
            )
            return {
                "success": False,
                "error":   "\n".join(filter_errors),
            }

    # Determine which fields to fetch: use metadata fieldnames, fall back to ["*"]
    if fields_meta:
        fetch_fields = [f["fieldname"] for f in fields_meta
                        if f["fieldtype"] not in (
                            "Text", "Long Text", "Small Text",
                            "Text Editor", "Code", "HTML Editor"
                        )]
        fetch_fields = ["name"] + [fn for fn in fetch_fields if fn != "name"][:15]
    else:
        fetch_fields = ["name"]

    try:
        records = frappe.get_all(
            doctype,
            filters=filters,
            fields=fetch_fields,
            limit_page_length=limit,
        )

        frappe.logger().info(
            "AI Assistant — SEARCH {} returned {} records".format(doctype, len(records))
        )

        # Build the display_fields list for the frontend card renderer
        if fields_meta:
            display_fields = [
                {"fieldname": f["fieldname"], "label": f["label"], "fieldtype": f["fieldtype"]}
                for f in fields_meta
                if f["fieldname"] in fetch_fields and f["fieldname"] != "name"
            ][:8]
        else:
            display_fields = []

        return {
            "success":        True,
            "action":         "search_documents",
            "doctype":        doctype,
            "filters":        filters,
            "count":          len(records),
            "records":        [dict(r) for r in records],
            "display_fields": display_fields,
            "message":        "Found {} {} record(s) matching your query".format(
                                len(records), doctype
                            ),
        }

    except frappe.DoesNotExistError:
        return {"success": False, "error": "DocType '{}' does not exist.".format(doctype)}
    except Exception as e:
        frappe.log_error(
            frappe.get_traceback(),
            "AI Assistant — SEARCH error [{}]".format(doctype)
        )
        return {"success": False, "error": "Failed to search {}: {}".format(doctype, str(e))}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 — Analytics & Aggregation Engine
# ─────────────────────────────────────────────────────────────────────────────

def handle_aggregate_documents(parsed, fields_meta=None):
    """
    Perform a count (or future numeric aggregate) on a DocType.

    Expected keys: doctype, operation ("count"), filters (optional)
    """
    doctype   = parsed["doctype"]
    operation = (parsed.get("operation") or "count").lower().strip()
    filters   = parsed.get("filters") or {}

    frappe.logger().info(
        "AI Assistant — AGGREGATE {} op={} filters={}".format(
            doctype, operation, json.dumps(filters)
        )
    )

    # Validate filter fields if we have metadata
    if fields_meta and filters:
        from frappe_assistant_bot.api.ai import _validate_search_filters
        errs = _validate_search_filters(filters, fields_meta)
        if errs:
            return {"success": False, "error": "\n".join(errs)}

    if operation != "count":
        return {
            "success": False,
            "error": "Unsupported aggregate operation '{}'. Only 'count' is currently supported.".format(operation),
        }

    try:
        count = frappe.db.count(doctype, filters=filters or None)

        # Build human-readable summary
        if filters:
            filter_desc = ", ".join(
                "{} = {}".format(k, v) for k, v in filters.items()
            )
            summary = "{:,} {} record(s) match the filter: {}.".format(
                count, doctype, filter_desc
            )
        else:
            summary = "There are {:,} {} record(s) in total.".format(count, doctype)

        frappe.logger().info("AI Assistant — AGGREGATE result: {}".format(count))
        return {
            "success":   True,
            "action":    "aggregate_documents",
            "doctype":   doctype,
            "operation": operation,
            "filters":   filters,
            "result":    count,
            "message":   summary,
        }

    except frappe.DoesNotExistError:
        return {"success": False, "error": "DocType '{}' does not exist.".format(doctype)}
    except Exception as e:
        frappe.log_error(
            frappe.get_traceback(),
            "AI Assistant — AGGREGATE error [{}]".format(doctype)
        )
        return {"success": False, "error": "Failed to aggregate {}: {}".format(doctype, str(e))}


def handle_group_documents(parsed, fields_meta=None):
    """
    Group documents by a field and return counts.

    Expected keys: doctype, group_by, metric ("count"), limit (optional)
    """
    doctype  = parsed["doctype"]
    group_by = parsed.get("group_by", "")
    metric   = (parsed.get("metric") or "count").lower().strip()
    limit    = int(parsed.get("limit", 10))
    limit    = min(limit, 50)

    if not group_by:
        return {"success": False, "error": "group_by field is required for group_documents."}

    # Validate group_by field against metadata
    if fields_meta:
        valid_fields = {f["fieldname"] for f in fields_meta}
        if group_by not in valid_fields:
            return {
                "success": False,
                "error": "Unknown group_by field '{}' — not present in DocType metadata.".format(group_by),
            }

    if metric != "count":
        return {
            "success": False,
            "error": "Unsupported metric '{}'. Only 'count' is currently supported.".format(metric),
        }

    frappe.logger().info(
        "AI Assistant — GROUP {} by {} metric={} limit={}".format(
            doctype, group_by, metric, limit
        )
    )

    try:
        # Use frappe.get_all with group_by for a clean ORM approach
        rows = frappe.get_all(
            doctype,
            fields=[group_by, "count(name) as total"],
            group_by=group_by,
            order_by="total desc",
            limit_page_length=limit,
        )

        # Build human-readable summary
        if rows:
            top  = rows[0]
            top_val   = top.get(group_by) or "(empty)"
            top_count = top.get("total", 0)
            summary = (
                "'{}' has the highest count with {:,} record(s). "
                "Showing top {} groups by {}."
            ).format(top_val, top_count, len(rows), group_by)
        else:
            summary = "No records found for {} grouped by {}.".format(doctype, group_by)

        frappe.logger().info(
            "AI Assistant — GROUP {} by {} returned {} groups".format(
                doctype, group_by, len(rows)
            )
        )
        return {
            "success":  True,
            "action":   "group_documents",
            "doctype":  doctype,
            "group_by": group_by,
            "metric":   metric,
            "rows":     [dict(r) for r in rows],
            "message":  summary,
        }

    except frappe.DoesNotExistError:
        return {"success": False, "error": "DocType '{}' does not exist.".format(doctype)}
    except Exception as e:
        frappe.log_error(
            frappe.get_traceback(),
            "AI Assistant — GROUP error [{}]".format(doctype)
        )
        return {"success": False, "error": "Failed to group {}: {}".format(doctype, str(e))}


# ── Central Action Dispatcher ─────────────────────────────────────────────────
ACTION_HANDLERS = {
    "create_document":    handle_create_document,
    "update_document":    handle_update_document,
    "delete_document":    handle_delete_document,
    "get_document":       handle_get_document,
    "list_documents":     handle_list_documents,
    "search_documents":   handle_search_documents,
    "aggregate_documents": handle_aggregate_documents,
    "group_documents":    handle_group_documents,
}


# ─────────────────────────────────────────────────────────────────────────────
# Whitelisted API Entry Point
# ─────────────────────────────────────────────────────────────────────────────

@frappe.whitelist()
def process_prompt(prompt):
    """
    Phase 9 (Performance Optimized) — Single-Pass AI Assistant.
    """
    t_start = time.perf_counter()
    
    # Initialize all perf slots
    frappe.flags.perf_times = {
        "normalization_gen": 0.0,
        "metadata": 0.0,
        "resolve": 0.0,
        "db": 0.0,
        "format": 0.0,
        "validation": 0.0
    }
    
    t_norm_gen = 0.0
    t_metadata = 0.0
    t_resolve = 0.0
    t_db = 0.0
    t_format = 0.0
    t_validation = 0.0
    
    if not prompt or not prompt.strip():
        return {"success": False, "error": "Prompt cannot be empty."}

    prompt_clean = prompt.strip()
    frappe.logger().info("AI Assistant — original prompt: {}".format(prompt_clean))

    # ── Step 0: Fast Path Detection ─────────────────────────────────────────
    fast_path_res = _check_fast_path(prompt_clean)
    if fast_path_res:
        frappe.logger().info("AI Assistant — fast path triggered for: {}".format(prompt_clean))
        action = fast_path_res["action"]
        doctype = fast_path_res["doctype"]
        
        # Load metadata (cached)
        t_meta_start = time.perf_counter()
        fields_for_doctype = get_doctype_metadata(doctype)
        t_metadata += time.perf_counter() - t_meta_start
        frappe.flags.perf_times["metadata"] += t_metadata
        
        t_db_start = time.perf_counter()
        handler = ACTION_HANDLERS.get(action)
        if action in ("search_documents", "aggregate_documents", "group_documents"):
            result = handler(fast_path_res, fields_meta=fields_for_doctype)
        else:
            result = handler(fast_path_res)
        t_db += time.perf_counter() - t_db_start
        frappe.flags.perf_times["db"] += t_db
        
        t_format_start = time.perf_counter()
        result["normalized_prompt"] = fast_path_res["normalized_prompt"]
        result["confidence"] = 1.0
        t_format += time.perf_counter() - t_format_start
        frappe.flags.perf_times["format"] += t_format
        
        t_total = time.perf_counter() - t_start
        _log_perf(
            t_total,
            frappe.flags.perf_times["normalization_gen"],
            frappe.flags.perf_times["metadata"],
            frappe.flags.perf_times["resolve"],
            frappe.flags.perf_times["db"],
            frappe.flags.perf_times["format"],
            frappe.flags.perf_times["validation"]
        )
        return result

    # ── Step 1: Candidate DocType Selection & Metadata Pre-fetching ──────────
    t_meta_start = time.perf_counter()
    available_doctypes = get_available_doctypes()
    candidates = _get_candidate_doctypes(prompt_clean, available_doctypes)
    fields_for_doctype = []
    doctype_for_metadata = candidates[0] if candidates else None
    if doctype_for_metadata:
        fields_for_doctype = get_doctype_metadata(doctype_for_metadata)
    t_metadata += time.perf_counter() - t_meta_start
    frappe.flags.perf_times["metadata"] += t_metadata

    # ── Step 2: Build system prompt and call Ollama in one single pass ───────
    t_llm_start = time.perf_counter()
    system_prompt = _build_system_prompt(candidates)
    full_prompt = "{}\n\nUser: {}\nResponse:".format(system_prompt, prompt_clean)

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
        },
        "keep_alive": "60m"
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data     = resp.json()
        raw_text = _strip_think_tags(data.get("response", ""))
        t_norm_gen = time.perf_counter() - t_llm_start
        frappe.flags.perf_times["normalization_gen"] += t_norm_gen

        if not raw_text:
            return {"success": False, "error": "Model returned an empty response."}

        # ── Step 3: Parse JSON ──────────────────────────────────────────────
        t_val_start = time.perf_counter()
        parsed = None
        parse_err = None
        try:
            parsed = _extract_json(raw_text)
        except ValueError as e:
            parse_err = str(e)

        if parsed is None:
            frappe.log_error(
                "Raw model output:\n{}".format(raw_text),
                "AI Assistant — JSON parse failure"
            )
            debug_msg = (
                "=== AI DEBUG ===\n"
                "Prompt:\n{}\n\n"
                "Raw Output:\n{}\n\n"
                "Parsed:\nFailed to parse JSON\n\n"
                "Validation:\n{}\n"
                "=== END ===".format(prompt_clean, raw_text, parse_err or "Invalid JSON structure")
            )
            print(debug_msg)
            frappe.logger().warning(debug_msg)
            return {
                "success": False,
                "error":   "Model returned invalid JSON.",
                "raw":     raw_text,
            }

        # Extract normalized prompt and confidence
        confidence = float(parsed.get("confidence", 1.0))
        normalized = parsed.get("normalized_prompt") or parsed.get("normalized") or prompt_clean
        suggestions = list(parsed.get("suggestions") or [])
        confidence = max(0.0, min(1.0, confidence))
        t_validation = time.perf_counter() - t_val_start
        frappe.flags.perf_times["validation"] += t_validation

        frappe.logger().info(
            "AI Assistant — single pass result: '{}' | confidence: {}".format(
                normalized, confidence
            )
        )

        # If model is not confident enough, return clarification suggestions
        if confidence < CONFIDENCE_THRESHOLD:
            # Print low-confidence debug info
            debug_msg = (
                "=== AI DEBUG ===\n"
                "Prompt:\n{}\n\n"
                "Raw Output:\n{}\n\n"
                "Parsed:\n{}\n\n"
                "Validation:\nLow Confidence ({:.2f} < {})\n"
                "=== END ===".format(
                    prompt_clean,
                    raw_text,
                    json.dumps(parsed, indent=2),
                    confidence,
                    CONFIDENCE_THRESHOLD
                )
            )
            print(debug_msg)
            frappe.logger().warning(debug_msg)

            frappe.logger().warning(
                "AI Assistant — low confidence ({:.2f}), returning suggestions.".format(confidence)
            )
            t_total = time.perf_counter() - t_start
            _log_perf(
                t_total,
                frappe.flags.perf_times["normalization_gen"],
                frappe.flags.perf_times["metadata"],
                frappe.flags.perf_times["resolve"],
                frappe.flags.perf_times["db"],
                frappe.flags.perf_times["format"],
                frappe.flags.perf_times["validation"]
            )
            return {
                "success":            False,
                "needs_clarification": True,
                "clarification_type": "low_confidence",
                "confidence":          confidence,
                "original_prompt":     prompt_clean,
                "suggestions":         suggestions or [
                    "Show all records",
                    "How many records exist?",
                    "Create a new record",
                ],
                "message": (
                    "I'm not sure what you mean. "
                    "Did you mean one of these?"
                ),
            }

        # ── Step 4: Validate action schema ─────────────────────────────────
        t_val_start = time.perf_counter()
        validation_error = None
        try:
            _validate_action(parsed)
        except ValueError as ve:
            validation_error = str(ve)
            frappe.flags.perf_times["validation"] += (time.perf_counter() - t_val_start)
            debug_msg = (
                "=== AI DEBUG ===\n"
                "Prompt:\n{}\n\n"
                "Raw Output:\n{}\n\n"
                "Parsed:\n{}\n\n"
                "Validation:\n{}\n"
                "=== END ===".format(
                    prompt_clean,
                    raw_text,
                    json.dumps(parsed, indent=2),
                    validation_error
                )
            )
            print(debug_msg)
            frappe.logger().warning(debug_msg)
            return {
                "success": False,
                "error":   "AI returned JSON but action schema validation failed.\n{}".format(validation_error),
                "raw":     raw_text,
            }

        # Print success debug logging
        debug_msg = (
            "=== AI DEBUG ===\n"
            "Prompt:\n{}\n\n"
            "Raw Output:\n{}\n\n"
            "Parsed:\n{}\n\n"
            "Validation:\nPassed\n\n"
            "Final Action:\n{}\n"
            "=== END ===".format(
                prompt_clean,
                raw_text,
                json.dumps(parsed, indent=2),
                parsed.get("action")
            )
        )
        print(debug_msg)
        frappe.logger().warning(debug_msg)

        # ── Step 5: Validate fields against DocType metadata ───────────────
        action = parsed["action"]
        doctype = parsed.get("doctype")
        
        # If the LLM returned a different doctype than the pre-fetched candidates,
        # fetch the metadata for the correct doctype
        if doctype and doctype not in candidates:
            t_meta_start2 = time.perf_counter()
            fields_for_doctype = get_doctype_metadata(doctype)
            frappe.flags.perf_times["metadata"] += (time.perf_counter() - t_meta_start2)

        if action in ("create_document", "update_document") and fields_for_doctype:
            field_errors = _validate_fields_against_metadata(
                parsed.get("data", {}), fields_for_doctype
            )
            if field_errors:
                frappe.flags.perf_times["validation"] += (time.perf_counter() - t_val_start)
                frappe.logger().warning(
                    "AI Assistant — field validation errors: {}".format(field_errors)
                )
                return {
                    "success": False,
                    "error":   "\n".join(field_errors),
                    "raw":     raw_text,
                }
        frappe.flags.perf_times["validation"] += (time.perf_counter() - t_val_start)

        # ── Step 6: Entity Resolution and Dispatch ─────────────────────────
        RESOLVE_ACTIONS = {"update_document", "delete_document", "get_document"}
        META_ACTIONS = {"search_documents", "aggregate_documents", "group_documents"}

        result = None
        resolve_time_before = frappe.flags.perf_times["resolve"]

        if action in RESOLVE_ACTIONS and parsed.get("name"):
            t_res_start = time.perf_counter()
            resolution = resolve_document(doctype, parsed["name"], fields_for_doctype)
            t_res_elapsed = time.perf_counter() - t_res_start

            if "needs_clarification" in resolution:
                t_total = time.perf_counter() - t_start
                _log_perf(
                    t_total,
                    frappe.flags.perf_times["normalization_gen"],
                    frappe.flags.perf_times["metadata"],
                    frappe.flags.perf_times["resolve"] + t_res_elapsed,
                    frappe.flags.perf_times["db"],
                    frappe.flags.perf_times["format"],
                    frappe.flags.perf_times["validation"]
                )
                return resolution

            if "error" in resolution:
                return {"success": False, "error": resolution["error"]}

            # Combine confidence!
            res_conf = resolution.get("confidence", 1.0)
            final_confidence = (confidence + res_conf) / 2.0
            frappe.logger().info("Combined confidence: LLM={:.2f}, Resolution={:.2f} -> Final={:.2f}".format(confidence, res_conf, final_confidence))

            if final_confidence < CONFIDENCE_THRESHOLD:
                t_total = time.perf_counter() - t_start
                _log_perf(
                    t_total,
                    frappe.flags.perf_times["normalization_gen"],
                    frappe.flags.perf_times["metadata"],
                    frappe.flags.perf_times["resolve"] + t_res_elapsed,
                    frappe.flags.perf_times["db"],
                    frappe.flags.perf_times["format"],
                    frappe.flags.perf_times["validation"]
                )
                return {
                    "success":            False,
                    "needs_clarification": True,
                    "clarification_type": "low_confidence",
                    "confidence":          final_confidence,
                    "original_prompt":     prompt_clean,
                    "suggestions":         suggestions or [
                        "Show all records",
                        "How many records exist?",
                        "Create a new record",
                    ],
                    "message": (
                        "I'm not sure what you mean. "
                        "Did you mean one of these?"
                    ),
                }

            # Override parsed["name"] with resolved name for execution
            parsed["name"] = resolution["resolved_name"]

            t_db_start = time.perf_counter()
            handler = ACTION_HANDLERS.get(action)
            result = handler(parsed, fields_meta=fields_for_doctype)
            t_db_elapsed = time.perf_counter() - t_db_start

            resolve_spent_in_handler = frappe.flags.perf_times["resolve"] - resolve_time_before
            actual_db_time = max(0.0, t_db_elapsed - resolve_spent_in_handler)
            frappe.flags.perf_times["db"] += actual_db_time

        elif action in META_ACTIONS:
            t_db_start = time.perf_counter()
            handler = ACTION_HANDLERS.get(action)
            result = handler(parsed, fields_meta=fields_for_doctype)
            t_db_elapsed = time.perf_counter() - t_db_start

            resolve_spent_in_handler = frappe.flags.perf_times["resolve"] - resolve_time_before
            actual_db_time = max(0.0, t_db_elapsed - resolve_spent_in_handler)
            frappe.flags.perf_times["db"] += actual_db_time

        else:
            t_db_start = time.perf_counter()
            handler = ACTION_HANDLERS.get(action)
            result = handler(parsed)
            t_db_elapsed = time.perf_counter() - t_db_start

            resolve_spent_in_handler = frappe.flags.perf_times["resolve"] - resolve_time_before
            actual_db_time = max(0.0, t_db_elapsed - resolve_spent_in_handler)
            frappe.flags.perf_times["db"] += actual_db_time

        # ── Step 7: Format output ──────────────────────────────────────────
        t_format_start = time.perf_counter()
        if result:
            result["normalized_prompt"] = normalized
            result["confidence"] = confidence
        t_format = time.perf_counter() - t_format_start
        frappe.flags.perf_times["format"] += t_format

        t_total = time.perf_counter() - t_start
        _log_perf(
            t_total,
            frappe.flags.perf_times["normalization_gen"],
            frappe.flags.perf_times["metadata"],
            frappe.flags.perf_times["resolve"],
            frappe.flags.perf_times["db"],
            frappe.flags.perf_times["format"],
            frappe.flags.perf_times["validation"]
        )
        return result

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": (
                "Could not connect to Ollama. "
                "Make sure Ollama is running on localhost:11434 "
                "and the model '{}' is pulled.".format(OLLAMA_MODEL)
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Ollama took too long to respond (timeout: {}s). Try a shorter prompt.".format(TIMEOUT),
        }
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "error": "Ollama returned an HTTP error: {}".format(str(e)),
        }
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "AI Assistant — process_prompt error")
        return {"success": False, "error": "Unexpected error: {}".format(str(e))}




# ───────────────────────────────────────────────────────────────────────────────
# FAC ASSISTANT  —  process_prompt_fac()
# ───────────────────────────────────────────────────────────────────────────────
# Full-intelligence assistant: fast-path, metadata injection, confidence scoring,
# entity resolution, clarification — execution via FAC tools ONLY.
# Legacy process_prompt() + ACTION_HANDLERS kept untouched for rollback.
# ───────────────────────────────────────────────────────────────────────────────

from frappe_assistant_bot.api.fac_layer import (
    _FAC_IMPORT_OK,
    _FAC_IMPORT_ERROR,
    FAC_TOOLS as _FAC_TOOLS,
    execute_fac_tool
)

# ── FAC-mode system prompt ─────────────────────────────────────────────────────
_FAC_BASE_INSTRUCTION = """You are a highly intelligent ERPNext AI assistant.
Read the user message (which may have typos, abbreviations, or indirect business references) and map it to the correct FAC tool and arguments in a single JSON object.

Return ONLY valid JSON. No markdown, no explanation, no code fences.

JSON Response Format:
{
  "normalized_prompt": "<clean restatement of intent>",
  "confidence": <float 0.0-1.0>,
  "suggestions": ["alt phrasing 1", "alt phrasing 2"],
  "tool": "<one of: list_documents, get_document, create_document, update_document, delete_document, search_documents, aggregate_documents, group_documents, report_list, generate_report, run_workflow, get_doctype_info>",
  "doctype": "<detected DocType>",
  "name": "<document name/identifier for get/update/delete>",
  "data": {<field:value pairs for create/update>},
  "filters": {<field:value filters for search/aggregate>},
  "operation": "<count for aggregate_documents>",
  "group_by": "<fieldname for group_documents>",
  "report_name": "<name for generate_report>",
  "workflow_action": "<Approve/Reject/Submit etc for run_workflow>",
  "limit": 20
}

CRITICAL RULES:
- You are a TOOL SELECTOR. Never return records, data, explanations, or answers.
- The backend executes the tool. Your ONLY job: pick the tool and fill arguments.
- Returning actual ERP records is FORBIDDEN.
- Returning natural language explanations is FORBIDDEN.

TOOL MAPPING:
- "show/list/get all X" → tool: list_documents
- "show X named Y / get X Y" → tool: get_document
- "create/add X" → tool: create_document
- "update/change/set X" → tool: update_document
- "delete/remove X" → tool: delete_document
- "search/find X where..." → tool: search_documents
- "how many X / count X" → tool: aggregate_documents, operation: count
- "group X by Y / X per Y" → tool: group_documents
- "sales report / inventory report" → tool: report_list or generate_report
- "approve / submit / reject X" → tool: run_workflow
- "what fields does X have" → tool: get_doctype_info

Examples:
User: List all customers
{"normalized_prompt":"List all Customer records","confidence":0.99,"suggestions":[],"tool":"list_documents","doctype":"Customer","limit":20,"name":"","data":{},"filters":{},"operation":"","group_by":"","report_name":"","workflow_action":""}

User: How many customers do we have
{"normalized_prompt":"Count Customer records","confidence":0.98,"suggestions":[],"tool":"aggregate_documents","doctype":"Customer","operation":"count","limit":20,"name":"","data":{},"filters":{},"group_by":"","report_name":"","workflow_action":""}

User: Delete kafka book
{"normalized_prompt":"Delete Book by Kafka","confidence":0.85,"suggestions":[],"tool":"delete_document","doctype":"Book","name":"kafka","limit":20,"data":{},"filters":{},"operation":"","group_by":"","report_name":"","workflow_action":""}
"""


def _fac_build_system_prompt(candidates=None):
    """Compose FAC system prompt with optional DocType metadata blocks."""
    prompt = _FAC_BASE_INSTRUCTION
    current_date = frappe.utils.today()
    prompt += "\n\nCURRENT DATE: {}\n".format(current_date)
    if candidates:
        for dt in candidates:
            fields = get_doctype_metadata(dt)
            prompt += _build_metadata_block(dt, fields)
    return prompt


def _fac_log(prompt, normalized, confidence, tool, arguments, t_llm, t_exec):
    msg = (
        "\n=== FAC EXECUTION ===\n"
        "Prompt:\n{}\n\n"
        "Normalized:\n{}\n\n"
        "Confidence:\n{:.2f}\n\n"
        "Tool:\n{}\n\n"
        "Arguments:\n{}\n\n"
        "LLM Time:  {:.3f}s\n"
        "Exec Time: {:.3f}s\n"
        "=== END ===\n"
    ).format(prompt, normalized, confidence, tool, json.dumps(arguments, indent=2), t_llm, t_exec)
    print(msg)
    frappe.logger().info(msg)


@frappe.whitelist()
def process_prompt_fac(prompt):
    """
    FAC Assistant — full intelligence, FAC execution only.

    Pipeline:
        Prompt → Fast-path → Candidate metadata → Qwen (single pass)
        → Confidence check → Entity resolution → FAC tool execution → Result
    """
    t_start = time.perf_counter()

    if not prompt or not prompt.strip():
        return {"success": False, "error": "Prompt cannot be empty."}

    prompt_clean = prompt.strip()
    frappe.logger().info("FAC Assistant — prompt: {}".format(prompt_clean))

    if not _FAC_IMPORT_OK:
        return {"success": False, "error": "FAC tool registry failed to load: {}".format(_FAC_IMPORT_ERROR)}

    # ── Step 0: Fast path ─────────────────────────────────────────────────────
    fast = _check_fast_path(prompt_clean)
    if fast:
        action  = fast["action"]
        doctype = fast["doctype"]
        t_db = time.perf_counter()
        # Build kwargs from fast-path result
        kwargs = {"doctype": doctype, "_prompt": prompt_clean}
        if fast.get("filters"):
            kwargs["filters"] = fast["filters"]
        if fast.get("limit"):
            kwargs["limit"] = fast["limit"]
        if fast.get("operation"):
            kwargs["operation"] = fast["operation"]
        result = execute_fac_tool(action, kwargs)
        t_exec = time.perf_counter() - t_db
        if result:
            result["normalized_prompt"] = fast["normalized_prompt"]
            result["confidence"] = 1.0
            result["fac_mode"] = True
        _fac_log(prompt_clean, fast["normalized_prompt"], 1.0, action, kwargs, 0.0, t_exec)
        return result

    # ── Step 1: Candidate DocTypes + metadata ─────────────────────────────────
    available   = get_available_doctypes()
    candidates  = _get_candidate_doctypes(prompt_clean, available)
    fields_meta = get_doctype_metadata(candidates[0]) if candidates else []

    # ── Step 2: Build prompt + call Ollama ────────────────────────────────────
    system_prompt = _fac_build_system_prompt(candidates)
    full_prompt   = "{}\n\nUser: {}\nResponse:".format(system_prompt, prompt_clean)

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 256},
        "keep_alive": "60m",
    }

    try:
        resp     = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        raw_text = _strip_think_tags(resp.json().get("response", ""))
        t_llm    = time.perf_counter() - t_start
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to Ollama on localhost:11434."}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Ollama timed out."}
    except Exception as e:
        return {"success": False, "error": "Ollama error: {}".format(str(e))}

    if not raw_text:
        return {"success": False, "error": "Model returned an empty response."}

    # ── Step 3: Parse JSON ────────────────────────────────────────────────────
    try:
        parsed = _extract_json(raw_text)
    except ValueError:
        parsed = None

    if parsed is None:
        return {"success": False, "error": "Model returned invalid JSON.", "raw": raw_text}

    confidence  = max(0.0, min(1.0, float(parsed.get("confidence", 1.0))))
    normalized  = parsed.get("normalized_prompt") or prompt_clean
    suggestions = list(parsed.get("suggestions") or [])
    tool        = parsed.get("tool", "")
    doctype     = parsed.get("doctype", "")

    # ── Step 4: Confidence gate ───────────────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "success": False, "needs_clarification": True,
            "clarification_type": "low_confidence",
            "confidence": confidence, "original_prompt": prompt_clean,
            "suggestions": suggestions or ["Show all records", "How many records exist?", "Create a new record"],
            "message": "I'm not sure what you mean. Did you mean one of these?",
        }

    # ── Step 5: Validate tool ─────────────────────────────────────────────────
    if not tool or tool not in _FAC_TOOLS:
        return {"success": False, "error": "that much functionality is not available in this phase", "raw": raw_text}

    # Refresh metadata if doctype differs from pre-fetched candidate
    if doctype and (not candidates or doctype not in candidates):
        fields_meta = get_doctype_metadata(doctype)

    # ── Step 6: Entity Resolution (for get/update/delete) ────────────────────
    RESOLVE_TOOLS = {"update_document", "delete_document", "get_document"}
    name = parsed.get("name", "")

    if tool in RESOLVE_TOOLS and name:
        resolution = resolve_document(doctype, name, fields_meta)

        if "needs_clarification" in resolution:
            return resolution

        if "error" in resolution:
            return {"success": False, "error": resolution["error"]}

        res_conf       = resolution.get("confidence", 1.0)
        final_conf     = (confidence + res_conf) / 2.0
        if final_conf < CONFIDENCE_THRESHOLD:
            return {
                "success": False, "needs_clarification": True,
                "clarification_type": "low_confidence",
                "confidence": final_conf, "original_prompt": prompt_clean,
                "suggestions": suggestions or ["Show all records", "Create a new record"],
                "message": "I'm not sure what you mean. Did you mean one of these?",
            }
        name = resolution["resolved_name"]

    # ── Step 7: Build FAC tool arguments ──────────────────────────────────────
    kwargs = {"doctype": doctype} if doctype else {}
    kwargs["_prompt"] = prompt_clean

    if tool in ("get_document", "update_document", "delete_document"):
        kwargs["name"] = name
    if tool in ("create_document", "update_document"):
        kwargs["data"] = parsed.get("data") or {}
    if tool in ("list_documents", "search_documents"):
        if parsed.get("filters"):
            kwargs["filters"] = parsed["filters"]
        kwargs["limit"] = int(parsed.get("limit", 20))
    if tool == "aggregate_documents":
        kwargs["operation"] = parsed.get("operation", "count")
        if parsed.get("filters"):
            kwargs["filters"] = parsed["filters"]
    if tool == "group_documents":
        kwargs["group_by"] = parsed.get("group_by", "")
    if tool == "generate_report":
        kwargs = {"report_name": parsed.get("report_name", ""), "filters": parsed.get("filters") or {}, "_prompt": prompt_clean}
    if tool == "run_workflow":
        kwargs["name"]            = name
        kwargs["action"]          = parsed.get("workflow_action", "")
    if tool == "get_doctype_info":
        kwargs = {"doctype": doctype, "_prompt": prompt_clean}

    # ── Step 8: Execute via FAC tool ──────────────────────────────────────────
    t_exec_start = time.perf_counter()
    try:
        result = execute_fac_tool(tool, kwargs)
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "FAC Assistant — execution error")
        return {"success": False, "error": "Execution error: {}".format(str(e))}

    t_exec = time.perf_counter() - t_exec_start
    t_llm_only = t_llm - (time.perf_counter() - t_start - t_exec)  # approximate

    # ── Step 9: Annotate + log + return ───────────────────────────────────────
    if result:
        result["normalized_prompt"] = normalized
        result["confidence"]        = confidence
        result["fac_mode"]          = True

    _fac_log(prompt_clean, normalized, confidence, tool, kwargs, t_llm, t_exec)
    return result
