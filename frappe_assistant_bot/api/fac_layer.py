"""
FAC Execution Layer
===================
Uses REAL FAC tool classes from frappe_assistant_core.
No custom CRUD logic. No frappe.get_all / frappe.get_doc / doc.insert calls.
FAC tool classes are the ONLY execution mechanism.

Output normalizer maps FAC responses → the shape the frontend renderers expect.
"""

import frappe
import json

# ── Import real FAC tool classes ──────────────────────────────────────────────
# We import at module level so import errors surface immediately at startup.
try:
    from frappe_assistant_core.plugins.core.tools.list_documents   import DocumentList
    from frappe_assistant_core.plugins.core.tools.get_document     import DocumentGet
    from frappe_assistant_core.plugins.core.tools.create_document  import DocumentCreate
    from frappe_assistant_core.plugins.core.tools.update_document  import DocumentUpdate
    from frappe_assistant_core.plugins.core.tools.delete_document  import DocumentDelete
    from frappe_assistant_core.plugins.core.tools.search_documents import SearchDocuments
    from frappe_assistant_core.plugins.core.tools.report_list      import ReportList
    from frappe_assistant_core.plugins.core.tools.generate_report  import GenerateReport
    from frappe_assistant_core.plugins.core.tools.run_workflow     import RunWorkflow
    from frappe_assistant_core.plugins.core.tools.get_doctype_info import GetDoctypeInfo
    from frappe_assistant_core.plugins.core.tools.get_pending_approvals import GetPendingApprovals

    _FAC_IMPORT_OK = True
    _FAC_IMPORT_ERROR = None
    frappe.logger().info("FAC tool registry loaded successfully")
    print("FAC tool registry loaded successfully")

except ImportError as _e:
    _FAC_IMPORT_OK = False
    _FAC_IMPORT_ERROR = str(_e)
    frappe.logger().error("FAC tool registry FAILED to load: {}".format(_e))
    print("FAC tool registry FAILED to load: {}".format(_e))

    # Stub classes so the module doesn't crash at import time
    class _Stub:
        name = "stub"
        def execute(self, arguments):
            return {"success": False, "error": "FAC import failed: {}".format(_FAC_IMPORT_ERROR)}

    DocumentList = DocumentGet = DocumentCreate = DocumentUpdate = DocumentDelete = _Stub
    SearchDocuments = ReportList = GenerateReport = RunWorkflow = GetDoctypeInfo = _Stub
    GetPendingApprovals = _Stub


# ─────────────────────────────────────────────────────────────────────────────
# FAC Tool Class Registry
# ─────────────────────────────────────────────────────────────────────────────
FAC_TOOL_CLASSES = {
    "list_documents":      DocumentList,
    "get_document":        DocumentGet,
    "create_document":     DocumentCreate,
    "update_document":     DocumentUpdate,
    "delete_document":     DocumentDelete,
    "search_documents":    SearchDocuments,
    "report_list":         ReportList,
    "generate_report":     GenerateReport,
    "run_workflow":        RunWorkflow,
    "get_doctype_info":    GetDoctypeInfo,
    "get_pending_approvals": GetPendingApprovals,
    # aggregate / group handled by adapter below (FAC uses list_documents with filters)
    "aggregate_documents": DocumentList,
    "group_documents":     DocumentList,
}


# ─────────────────────────────────────────────────────────────────────────────
# Output Normalizer
# ─────────────────────────────────────────────────────────────────────────────
# FAC tools return their own shape. The frontend renderers expect a specific
# shape (action, records, count, message, …). This normalizer bridges the gap.

def _normalize(fac_result, action, doctype=""):
    """
    Map a raw FAC tool result → the shape expected by the frontend renderers.
    Preserves all FAC fields and adds the renderer-required keys.
    """
    if not isinstance(fac_result, dict):
        return {"success": False, "error": "FAC returned non-dict result", "action": action}

    out = dict(fac_result)          # copy — don't mutate FAC result
    out["action"] = action
    out["fac_class"] = FAC_TOOL_CLASSES.get(action, type(None)).__name__

    if not out.get("doctype") and doctype:
        out["doctype"] = doctype

    # list_documents / search_documents: FAC uses "data", frontend needs "records"
    if action in ("list_documents", "search_documents", "aggregate_documents"):
        raw_data = fac_result.get("data", [])
        if isinstance(raw_data, list):
            # frontend list renderer expects a flat list of name strings
            out["records"] = [
                r.get("name", str(r)) if isinstance(r, dict) else str(r)
                for r in raw_data
            ]
            out["count"] = len(raw_data)
        if out.get("success") and not out.get("message"):
            out["message"] = "Found {} {} record(s)".format(out.get("count", 0), doctype)

    # aggregate_documents: derive count from list result
    if action == "aggregate_documents":
        if out.get("success"):
            out["result"]    = fac_result.get("total_count") or fac_result.get("count") or len(fac_result.get("data", []))
            out["operation"] = "count"
            out["message"]   = "Total {} records: {}".format(doctype, out["result"])

    # get_document: FAC uses "data", frontend needs "document"
    if action == "get_document":
        if "data" in fac_result and "document" not in fac_result:
            out["document"] = fac_result["data"]
        if out.get("success") and not out.get("message"):
            out["message"] = "Fetched {} {}".format(doctype, fac_result.get("name", ""))

    # create_document
    if action == "create_document":
        out.setdefault("document_created", out.get("success", False))
        if out.get("success") and not out.get("message"):
            out["message"] = "{} {} created successfully".format(doctype, fac_result.get("name", ""))

    # update_document
    if action == "update_document":
        if out.get("success") and not out.get("message"):
            out["message"] = "{} {} updated successfully".format(doctype, fac_result.get("name", ""))
        if "updated_fields" not in out and "data" in fac_result:
            out["updated_fields"] = list(fac_result["data"].keys()) if isinstance(fac_result.get("data"), dict) else []

    # delete_document
    if action == "delete_document":
        if out.get("success") and not out.get("message"):
            out["message"] = "{} {} deleted successfully".format(doctype, fac_result.get("name", ""))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# FAC Tool Executor — single entry point
# ─────────────────────────────────────────────────────────────────────────────

def execute_fac_tool(tool_name, arguments, fac_class_name=None):
    """
    Instantiate the real FAC tool class and call .execute(arguments).

    Returns a normalized dict compatible with the frontend renderers.
    Logs the full execution details.
    """
    import time
    t0 = time.perf_counter()

    tool_class = FAC_TOOL_CLASSES.get(tool_name)
    if tool_class is None:
        return {
            "success": False,
            "error": "Unknown FAC tool: '{}'".format(tool_name),
            "action": tool_name,
        }

    class_name = tool_class.__name__

    # Special argument mapping for aggregate_documents (uses list_documents internally)
    exec_args = dict(arguments)
    if tool_name == "aggregate_documents":
        exec_args.setdefault("limit", 1)  # we only need the count, keep it light

    try:
        instance = tool_class()
        raw_result = instance.execute(exec_args)

        # Automatic validation-repair attempt for missing required child tables
        repaired_result = _attempt_validation_repair(tool_name, exec_args, raw_result)
        if repaired_result is not None:
            raw_result = repaired_result
    except Exception as e:
        frappe.log_error(frappe.get_traceback(), "FAC Tool Execution Error — {}".format(tool_name))
        return {
            "success": False,
            "error": "FAC tool '{}' raised an exception: {}".format(class_name, str(e)),
            "action": tool_name,
        }

    t_exec = time.perf_counter() - t0
    doctype = arguments.get("doctype", "")

    # Normalize output
    result = _normalize(raw_result, tool_name, doctype)
    result["fac_mode"] = True
    result["fac_class"] = class_name

    # Structured debug log
    _log_fac_execution(
        prompt=arguments.get("_prompt", ""),
        tool=tool_name,
        cls=class_name,
        args=exec_args,
        success=result.get("success", False),
        t_exec=t_exec,
    )

    return result


def _attempt_validation_repair(tool_name, exec_args, raw_result):
    """
    Attempts to repair missing child table fields automatically and retries execution.
    """
    if tool_name != "create_document":
        return None

    if not raw_result or raw_result.get("success"):
        return None

    # Check for missing required fields
    if raw_result.get("error_type") != "missing_required_field":
        return None

    missing_fields = raw_result.get("missing_fields")
    if not missing_fields:
        return None

    doctype = exec_args.get("doctype")
    if not doctype:
        return None

    # Get DocType info metadata
    try:
        from frappe_assistant_core.plugins.core.tools.get_doctype_info import GetDoctypeInfo
        info_instance = GetDoctypeInfo()
        info_res = info_instance.execute({"doctype": doctype})
    except Exception as e:
        frappe.logger().error("Validation Repair: Failed to get doctype info for {}: {}".format(doctype, str(e)))
        return None

    if not info_res or not info_res.get("success"):
        return None

    child_tables = info_res.get("child_tables", [])
    if not child_tables:
        return None

    parent_data = exec_args.setdefault("data", {})
    repaired_any = False

    for field_name in missing_fields:
        # Check if the missing field is a child table
        child_table = None
        for ct in child_tables:
            if ct.get("fieldname") == field_name:
                child_table = ct
                break

        if not child_table:
            continue

        # Found a missing child table! Let's build a minimal valid row.
        row = {}
        for cf in child_table.get("fields", []):
            cf_name = cf.get("fieldname")
            cf_type = cf.get("fieldtype")
            cf_opts = cf.get("options")
            cf_reqd = cf.get("reqd")

            # Try to populate linked documents or required fields
            if cf_reqd or cf_type == "Link":
                if cf_type == "Link" and cf_opts:
                    # Find an existing record of the linked DocType
                    val = frappe.db.get_value(cf_opts, {}, "name")
                    if not val:
                        recs = frappe.get_all(cf_opts, limit=1)
                        if recs:
                            val = recs[0].name
                    if val:
                        row[cf_name] = val
                elif cf_type == "Select" and cf_opts:
                    opts = [o.strip() for o in cf_opts.split("\n") if o.strip()]
                    if opts:
                        row[cf_name] = opts[0]
                elif cf_type in ("Int", "Float", "Currency", "Percent"):
                    row[cf_name] = 1
                elif cf_type in ("Date", "Datetime"):
                    row[cf_name] = frappe.utils.today()
                elif cf_type in ("Data", "Text", "Small Text", "Code"):
                    row[cf_name] = "Auto-populated"

        if row:
            parent_data[field_name] = [row]
            repaired_any = True
            frappe.logger().info("Validation Repair: Auto-populated child table '{}' with row: {}".format(field_name, row))

    if repaired_any:
        # Retry document creation once
        try:
            from frappe_assistant_core.plugins.core.tools.create_document import DocumentCreate
            retry_instance = DocumentCreate()
            retry_result = retry_instance.execute(exec_args)
            frappe.logger().info("Validation Repair: Retry result success={}".format(retry_result.get("success")))
            return retry_result
        except Exception as e:
            frappe.logger().error("Validation Repair: Retry failed: {}".format(str(e)))
            return None

    return None


def _log_fac_execution(prompt, tool, cls, args, success, t_exec):
    safe_args = {k: v for k, v in args.items() if k != "_prompt"}
    msg = (
        "\n=== FAC TOOL EXECUTION ===\n"
        "Prompt:\n{prompt}\n\n"
        "Tool:\n{tool}\n\n"
        "FAC Class:\n{cls}\n\n"
        "Arguments:\n{args}\n\n"
        "Execution Time:\n{t:.3f}s\n\n"
        "Success:\n{ok}\n"
        "=== END ===\n"
    ).format(
        prompt=prompt,
        tool=tool,
        cls=cls,
        args=json.dumps(safe_args, indent=2, default=str),
        t=t_exec,
        ok=success,
    )
    print(msg)
    frappe.logger().info(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Public registry (used by process_prompt_fac in ai.py)
# ─────────────────────────────────────────────────────────────────────────────
FAC_TOOLS = FAC_TOOL_CLASSES   # kept for backward compat with import in ai.py
