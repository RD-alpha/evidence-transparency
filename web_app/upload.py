import os
import uuid
import json
import shutil
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from flask import Blueprint, request, jsonify, render_template, current_app
from werkzeug.utils import secure_filename
from mistralai.client import Mistral
import pdf_text_extractor
from combine import CombineError, collect_sources, combinable_documents, combine_documents
from preprocess import (
    HarvardCitationExtractor,
    InTextCitationParser,
    MarkdownDocumentProcessor,
    _write_input,
    harvard_reference_alias_map,
)
from web_parser import GovUKParser, write_document_artifacts
from llm_monitor import track_llm_call

UPLOAD_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{32}$")
STALE_UPLOAD_MAX_AGE_SECONDS = 3600

upload_bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {"pdf"}
ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Client is created lazily so the API key is read at call-time,
# not at import-time (helps with testing/config flexibility).
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY environment variable is not set")
        _client = Mistral(api_key=api_key)
    return _client


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@upload_bp.route("/upload-page")
def upload_page():
    return render_template("upload.jinja")


def _cleanup_stale_uploads(upload_folder: Path) -> None:
    """Best-effort removal of temp files left behind by an analyze that was never confirmed."""
    now = time.time()
    for path in upload_folder.glob("*"):
        try:
            if path.is_file() and now - path.stat().st_mtime > STALE_UPLOAD_MAX_AGE_SECONDS:
                path.unlink()
        except OSError:
            pass


@upload_bp.route("/upload/analyze", methods=["POST"])
def upload_analyze():
    """Phase 1: save the PDF and extract its text layer -- no OCR, no cost.

    Reports which pages (if any) have too little extractable text and contain
    embedded images, i.e. look like scanned pages that would need OCR to recover.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    title = (request.form.get("title") or "").strip()
    alias = (request.form.get("alias") or "").strip()
    if not title:
        return jsonify({"error": "A document title is required"}), 400
    if not ALIAS_PATTERN.fullmatch(alias):
        return jsonify({"error": "Alias may contain only letters, numbers, hyphens, and underscores"}), 400

    documents_dir = Path(current_app.root_path).parent / "documents"
    if (documents_dir / alias).exists():
        return jsonify({"error": f"A document with alias '{alias}' already exists"}), 409

    filename = secure_filename(file.filename)
    upload_token = uuid.uuid4().hex
    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_uploads(upload_folder)

    temporary_path = upload_folder / f"{upload_token}_{filename}"
    file.save(temporary_path)

    try:
        extraction = pdf_text_extractor.analyze_pdf(str(temporary_path))
    except Exception as e:
        temporary_path.unlink(missing_ok=True)
        return jsonify({"error": f"Could not read PDF: {str(e)}"}), 500

    flagged_pages = [
        {"page": i + 1, "reason": page["ocr_reason"]}
        for i, page in enumerate(extraction["pages"])
        if page["needs_ocr"]
    ]

    return jsonify({
        "upload_token": upload_token,
        "filename": filename,
        "total_pages": extraction["page_count"],
        "flagged_pages": flagged_pages,
    })


@upload_bp.route("/upload/confirm", methods=["POST"])
def upload_confirm():
    """Phase 2: build the document, using OCR only for pages the user approved."""
    payload = request.get_json(silent=True) or request.form

    upload_token = (payload.get("upload_token") or "").strip()
    filename = (payload.get("filename") or "").strip()
    title = (payload.get("title") or "").strip()
    alias = (payload.get("alias") or "").strip()
    use_ocr_fallback = bool(payload.get("use_ocr_fallback"))

    if not UPLOAD_TOKEN_PATTERN.fullmatch(upload_token) or not filename:
        return jsonify({"error": "Missing or invalid upload session"}), 400
    if not title:
        return jsonify({"error": "A document title is required"}), 400
    if not ALIAS_PATTERN.fullmatch(alias):
        return jsonify({"error": "Alias may contain only letters, numbers, hyphens, and underscores"}), 400

    upload_folder = Path(current_app.config["UPLOAD_FOLDER"])
    temporary_path = upload_folder / f"{upload_token}_{filename}"
    if not temporary_path.exists():
        return jsonify({"error": "Upload session expired or not found. Please upload the file again."}), 404

    documents_dir = Path(current_app.root_path).parent / "documents"
    document_dir = documents_dir / alias
    if document_dir.exists():
        temporary_path.unlink(missing_ok=True)
        return jsonify({"error": f"A document with alias '{alias}' already exists"}), 409

    try:
        extraction = pdf_text_extractor.analyze_pdf(str(temporary_path))
    except Exception as e:
        temporary_path.unlink(missing_ok=True)
        return jsonify({"error": f"Could not read PDF: {str(e)}"}), 500

    if extraction["reference_gaps"] or extraction["reference_duplicates"]:
        current_app.logger.warning(
            "Reference numbering issue in '%s' (alias=%s): missing=%s duplicated=%s",
            filename, alias, extraction["reference_gaps"], extraction["reference_duplicates"],
        )

    flagged_indices = {i for i, page in enumerate(extraction["pages"]) if page["needs_ocr"]}

    ocr_response = None
    if use_ocr_fallback and flagged_indices:
        try:
            ocr_response = _call_mistral_ocr(str(temporary_path), alias)
        except Exception as e:
            temporary_path.unlink(missing_ok=True)
            return jsonify({"error": f"OCR failed: {str(e)}"}), 500

    all_blocks = []
    all_references = []
    seen_numbers = set()
    pages_via_ocr = []

    for i, page_extraction in enumerate(extraction["pages"]):
        if i in flagged_indices and ocr_response is not None:
            body, footer = _ocr_page_content(ocr_response.pages[i])
            page_blocks, page_references = pdf_text_extractor.merge_ocr_page(body, footer)
            pages_via_ocr.append(i + 1)
        else:
            page_blocks = page_extraction["blocks"]
            page_references = page_extraction["references"]

        all_blocks.extend(page_blocks)
        for ref in page_references:
            if ref["number"] not in seen_numbers:
                all_references.append(ref)
                seen_numbers.add(ref["number"])

    for ref in extraction.get("harvard_references", []):
        if ref["number"] not in seen_numbers:
            all_references.append(ref)
            seen_numbers.add(ref["number"])

    if not flagged_indices:
        extraction_method = "text_layer"
    elif len(pages_via_ocr) == extraction["page_count"]:
        extraction_method = "ocr"
    elif pages_via_ocr:
        extraction_method = "hybrid"
    else:
        extraction_method = "text_layer"

    try:
        document_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        temporary_path.unlink(missing_ok=True)
        return jsonify({"error": f"A document with alias '{alias}' already exists"}), 409

    citation_parser = None
    alias_map = harvard_reference_alias_map(extraction.get("harvard_references", []))
    if alias_map:
        citation_parser = InTextCitationParser()
        citation_parser.add_extractor(HarvardCitationExtractor(alias_map=alias_map))

    processor = MarkdownDocumentProcessor(citation_parser=citation_parser)
    references = processor.process_structured(all_blocks, all_references, document_dir)
    _write_input(document_dir)

    all_links = extraction.get("links", [])
    (document_dir / "links.json").write_text(
        json.dumps(all_links, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    uploaded_at = datetime.now().astimezone()
    metadata = {
        "title": title,
        "alias": alias,
        "uploaded_at": uploaded_at.isoformat(),
        "uploaded_date": uploaded_at.date().isoformat(),
        "uploaded_time": uploaded_at.strftime("%H:%M:%S %Z"),
        "extraction_method": extraction_method,
        "pages_via_ocr": pages_via_ocr,
    }
    try:
        (document_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(temporary_path), document_dir / "doc.pdf")
    except Exception as e:
        shutil.rmtree(document_dir, ignore_errors=True)
        temporary_path.unlink(missing_ok=True)
        return jsonify({"error": f"Could not save uploaded document: {str(e)}"}), 500

    preview_text = "\n\n".join(block["text"] for block in all_blocks)
    with open(document_dir / "input.txt", encoding="utf-8") as f:
        sentence_count = sum(1 for _ in f)

    return jsonify({
        "filename": filename,
        "alias": alias,
        "title": title,
        "text": preview_text,
        "extraction_method": extraction_method,
        "pages_via_ocr": pages_via_ocr,
        "reference_count": len(references),
        "link_count": len(all_links),
        "sentence_count": sentence_count,
    })


@upload_bp.route("/upload-url", methods=["POST"])
def upload_url():
    payload = request.get_json(silent=True) or request.form

    title = (payload.get("title") or "").strip()
    alias = (payload.get("alias") or "").strip()
    url = (payload.get("url") or "").strip()

    if not title:
        return jsonify({"error": "A document title is required"}), 400
    if not ALIAS_PATTERN.fullmatch(alias):
        return jsonify({"error": "Alias may contain only letters, numbers, hyphens, and underscores"}), 400
    if not url:
        return jsonify({"error": "A URL is required"}), 400

    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc.lower().endswith("gov.uk"):
        return jsonify({"error": "Please provide a gov.uk page URL"}), 400

    documents_dir = Path(current_app.root_path).parent / "documents"
    document_dir = documents_dir / alias
    if document_dir.exists():
        return jsonify({"error": f"A document with alias '{alias}' already exists"}), 409

    try:
        results = GovUKParser(url=url).parse()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch or parse the page: {str(e)}"}), 502

    if not results.get("govspeak_text_clean"):
        return jsonify({"error": "No content could be extracted from that page"}), 422

    try:
        document_dir.mkdir(parents=True, exist_ok=False)
        stats = write_document_artifacts(results, document_dir)
    except FileExistsError:
        return jsonify({"error": f"A document with alias '{alias}' already exists"}), 409
    except Exception as e:
        shutil.rmtree(document_dir, ignore_errors=True)
        return jsonify({"error": f"Could not process the page: {str(e)}"}), 500

    uploaded_at = datetime.now().astimezone()
    metadata = {
        "title": title,
        "alias": alias,
        "source_url": url,
        "uploaded_at": uploaded_at.isoformat(),
        "uploaded_date": uploaded_at.date().isoformat(),
        "uploaded_time": uploaded_at.strftime("%H:%M:%S %Z"),
    }
    try:
        (document_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        return jsonify({"error": f"Could not save document metadata: {str(e)}"}), 500

    return jsonify({
        "alias": alias,
        "title": title,
        "source_url": url,
        "sentence_count": stats["sentence_count"],
        "reference_count": stats["reference_count"],
        "link_count": stats["link_count"],
    })


@upload_bp.route("/upload/documents", methods=["GET"])
def upload_documents():
    """List the ingested documents that can be picked as combine sources."""
    documents_dir = Path(current_app.root_path).parent / "documents"
    return jsonify({"documents": combinable_documents(documents_dir)})


@upload_bp.route("/upload/combine", methods=["POST"])
def upload_combine():
    """Merge two or more ingested documents into a single combined document.

    Works entirely from the sources' persisted artefacts -- no extraction, no
    OCR, no LLM call, so no API cost.
    """
    payload = request.get_json(silent=True) or request.form

    title = (payload.get("title") or "").strip()
    alias = (payload.get("alias") or "").strip()
    sources = payload.get("sources")
    if isinstance(sources, str):
        sources = [part.strip() for part in sources.split(",") if part.strip()]
    sources = [str(source).strip() for source in (sources or [])]

    if not title:
        return jsonify({"error": "A document title is required"}), 400
    if not ALIAS_PATTERN.fullmatch(alias):
        return jsonify({"error": "Alias may contain only letters, numbers, hyphens, and underscores"}), 400
    if len(sources) < 2:
        return jsonify({"error": "Choose at least two documents to combine"}), 400
    if len(set(sources)) != len(sources):
        return jsonify({"error": "Each document can only be combined once"}), 400
    if any(not ALIAS_PATTERN.fullmatch(source) for source in sources):
        return jsonify({"error": "Source aliases may contain only letters, numbers, hyphens, and underscores"}), 400
    if alias in sources:
        return jsonify({"error": "The combined document needs an alias of its own"}), 400

    documents_dir = Path(current_app.root_path).parent / "documents"
    document_dir = documents_dir / alias
    if document_dir.exists():
        return jsonify({"error": f"A document with alias '{alias}' already exists"}), 409

    try:
        resolved_sources = collect_sources(documents_dir, sources)
    except CombineError as e:
        return jsonify({"error": str(e)}), 404

    try:
        document_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return jsonify({"error": f"A document with alias '{alias}' already exists"}), 409

    try:
        stats = combine_documents(resolved_sources, document_dir)
    except CombineError as e:
        shutil.rmtree(document_dir, ignore_errors=True)
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        shutil.rmtree(document_dir, ignore_errors=True)
        return jsonify({"error": f"Could not combine the documents: {str(e)}"}), 500

    combined_at = datetime.now().astimezone()
    metadata = {
        "title": title,
        "alias": alias,
        "uploaded_at": combined_at.isoformat(),
        "uploaded_date": combined_at.date().isoformat(),
        "uploaded_time": combined_at.strftime("%H:%M:%S %Z"),
        "extraction_method": "combined",
        "combined_from": stats["sources"],
    }
    try:
        (document_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        shutil.rmtree(document_dir, ignore_errors=True)
        return jsonify({"error": f"Could not save document metadata: {str(e)}"}), 500

    return jsonify({"alias": alias, "title": title, **stats})


def _call_mistral_ocr(filepath, alias=""):
    """Upload the PDF to Mistral, run OCR, and return the raw per-page response."""
    client = get_client()

    with track_llm_call("ocr", alias, "mistral", "mistral-ocr-latest") as call:
        with open(filepath, "rb") as f:
            uploaded = client.files.upload(
                file={
                    "file_name": os.path.basename(filepath),
                    "content": f,
                },
                purpose="ocr",
            )

        signed_url = client.files.get_signed_url(file_id=uploaded.id)

        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            },
            extract_header=True,
            extract_footer=True,
        )
        usage_info = getattr(ocr_response, "usage_info", None)
        pages_processed = getattr(usage_info, "pages_processed", None) if usage_info else None
        if pages_processed is None:
            pages_processed = len(ocr_response.pages)
        call.set_page_usage(pages_processed)

    return ocr_response


def _ocr_page_content(page):
    """Split one OCR'd page's blocks into (body_markdown, footer_text)."""
    references = []
    pages_text = []
    for block in page.blocks:
        if block.type in ["references", 'footer']:
            cleaned = block.content.strip()
            if len(cleaned) > 3:
                if cleaned[0] in "123456789⁰¹²³⁴⁵⁶⁷⁸⁹[#" or re.match(r'^\d+', cleaned):
                    references.append(cleaned)
        else:
            pages_text.append(block.content)
    return "\n\n".join(pages_text), "\n\n".join(references)
