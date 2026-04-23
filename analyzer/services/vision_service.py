"""Vision-AI document extraction for the AI chat RAG pipeline.

Given a Paperless doc_id + title, downloads the archived PDF, rasterises
each page to PNG, and calls the configured vision-capable LLM (via the
proxy pool with direct-provider fallback) to extract text from each page.

Used when a document's OCR content is empty or too short to be useful.

Extracted verbatim from `analyzer/routes/chat.py` during the 2026-04-23
maintainability refactor. No behavior change.
"""
import logging

logger = logging.getLogger(__name__)


def vision_extract_doc(doc_id: int, title: str, paperless_client, ai_config: dict) -> str:
    """
    Extract text from a Paperless document using Vision AI (GPT-4o or Claude Vision).
    Downloads the archived PDF, converts each page to a PNG, and sends to the configured
    LLM with vision capability.  Returns concatenated page text, or '' on any failure.
    Used during AI chat to enrich documents whose OCR content is empty or too short.
    """
    try:
        import base64
        from io import BytesIO

        pdf_bytes = paperless_client.download_document(doc_id, archived=True)
        if not pdf_bytes or len(pdf_bytes) < 200:
            return ''

        # Build list of base64-encoded page PNGs
        page_images = []
        try:
            from pdf2image import convert_from_bytes
            imgs = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=8)
            for img in imgs:
                buf = BytesIO()
                img.save(buf, format='PNG')
                page_images.append(base64.b64encode(buf.getvalue()).decode())
        except ImportError:
            # pdf2image not installed — treat bytes as a raw image
            page_images.append(base64.b64encode(pdf_bytes).decode())
        except Exception:
            page_images.append(base64.b64encode(pdf_bytes).decode())

        if not page_images:
            return ''

        vision_prompt = (
            "Extract ALL text from this document page accurately. "
            "Include all numbers, dates, names, headings, account numbers, addresses, "
            "and table data (format table columns separated by |). "
            "Output the raw extracted text only — no commentary."
        )

        # Pick the first enabled provider that has an API key
        provider_name, api_key = None, None
        for p in ai_config.get('chat', {}).get('providers', []):
            if p.get('enabled') and p.get('api_key', '').strip():
                provider_name = p['name']
                api_key = p['api_key'].strip()
                break

        if not provider_name or not api_key:
            return ''

        from analyzer.llm.proxy_call import call_llm, LLMUnavailableError
        extracted_pages = []
        for i, img_b64 in enumerate(page_images[:8]):
            try:
                # OpenAI-format multimodal message — proxy translates to anthropic
                # format upstream when needed. Direct-provider fallback via
                # direct_provider=... also uses OpenAI format (Anthropic fallback
                # for vision is handled only via proxy in practice).
                result = call_llm(
                    messages=[{'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img_b64}'}},
                        {'type': 'text', 'text': vision_prompt},
                    ]}],
                    task='extraction',
                    has_images=True,
                    max_tokens=2000,
                    operation='chat_vision_extract',
                    document_id=doc_id,
                    direct_provider=provider_name,
                    direct_api_key=api_key,
                    direct_model='gpt-4o' if provider_name == 'openai' else 'claude-3-5-sonnet-20241022',
                )
                page_text = result['content'] or ''
                if page_text:
                    extracted_pages.append(f"[Page {i + 1}]\n{page_text}")
            except (LLMUnavailableError, Exception) as page_err:
                logger.warning(f"Vision AI page {i + 1} failed for doc {doc_id}: {page_err}")

        result_text = '\n\n'.join(extracted_pages)
        logger.info(f"Vision AI extracted {len(result_text)} chars from doc {doc_id} ({len(page_images)} pages)")
        return result_text

    except Exception as e:
        logger.warning(f"vision_extract_doc failed for doc {doc_id} ({title}): {e}")
        return ''
