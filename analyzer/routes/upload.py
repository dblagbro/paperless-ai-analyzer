import logging
import os
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user

from analyzer.db import log_import, get_import_history
from analyzer.app import safe_json_body

logger = logging.getLogger(__name__)

bp = Blueprint('upload', __name__)

# Allowed file extensions for directory-scan (OCR-able or text-bearing documents)
_ALLOWED_SCAN_EXTS = {
    '.pdf',
    '.png', '.jpg', '.jpeg', '.gif', '.tiff', '.tif', '.bmp', '.webp', '.heic', '.heif',
    '.docx', '.doc', '.odt', '.rtf', '.txt', '.md', '.rst',
    '.xlsx', '.xls', '.ods', '.csv',
    '.pptx', '.ppt', '.odp',
    '.eml', '.msg',
    '.djvu', '.epub',
}


@bp.route('/api/upload/transform-url', methods=['POST'])
@login_required
def api_transform_upload_url():
    """Transform a cloud share link into a direct download URL."""
    import re as _re
    try:
        data = safe_json_body()
        raw_url = data.get('url', '').strip()
        if not raw_url:
            return jsonify({'error': 'url is required'}), 400

        service = 'generic'
        direct_url = raw_url
        filename_hint = None

        # Google Drive file
        m = _re.match(
            r'https://drive\.google\.com/file/d/([^/?#]+)',
            raw_url, _re.IGNORECASE
        )
        if m:
            file_id = m.group(1)
            direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
            service = 'google_drive'

        # Google Docs / Sheets / Slides
        if service == 'generic':
            m = _re.match(
                r'https://docs\.google\.com/(document|spreadsheets|presentation)/d/([^/?#]+)',
                raw_url, _re.IGNORECASE
            )
            if m:
                doc_type = m.group(1)
                file_id = m.group(2)
                fmt_map = {
                    'document': 'pdf',
                    'spreadsheets': 'pdf',
                    'presentation': 'pdf',
                }
                fmt = fmt_map.get(doc_type, 'pdf')
                direct_url = (
                    f'https://docs.google.com/{doc_type}/d/{file_id}/export?format={fmt}'
                )
                service = 'google_drive'
                filename_hint = f'document.{fmt}'

        # Dropbox
        if service == 'generic' and 'dropbox.com' in raw_url.lower():
            direct_url = _re.sub(r'[?&]dl=0', lambda m2: m2.group(0).replace('dl=0', 'dl=1'), raw_url)
            if 'dl=1' not in direct_url:
                sep = '&' if '?' in direct_url else '?'
                direct_url = direct_url + sep + 'dl=1'
            service = 'dropbox'

        # OneDrive 1drv.ms
        if service == 'generic' and '1drv.ms' in raw_url.lower():
            service = 'onedrive'

        logger.info(f"transform-url: {service} → {direct_url[:80]}")
        return jsonify({
            'direct_url': direct_url,
            'service': service,
            'filename_hint': filename_hint,
        })

    except Exception as e:
        logger.error(f"transform-url error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/upload/scan-url', methods=['POST'])
@login_required
def api_scan_url():
    """Probe a URL: return single-file info OR a list of compatible file links found on an
    HTML directory-listing page."""
    import re as _re
    from urllib.parse import urljoin, urlparse as _urlparse, unquote as _unquote
    try:
        data = safe_json_body()
        url = data.get('url', '').strip()
        auth_type = data.get('auth_type', 'none')
        username = data.get('username')
        password = data.get('password')
        token = data.get('token')

        if not url:
            return jsonify({'error': 'url is required'}), 400

        import requests as _req

        session_auth = None
        req_headers = {'User-Agent': 'Paperless-AI-Analyzer/2.0'}
        if auth_type == 'basic' and username and password:
            session_auth = (username, password)
        elif auth_type == 'token' and token:
            req_headers['Authorization'] = f'Bearer {token}'

        try:
            head = _req.head(url, auth=session_auth, headers=req_headers,
                             allow_redirects=True, timeout=15)
            content_type = head.headers.get('content-type', '').lower().split(';')[0].strip()
        except Exception as e:
            return jsonify({'error': f'Could not reach URL: {e}'}), 400

        path = _urlparse(url).path
        m = _re.search(r'(\.[a-z0-9]{2,6})(?:[?#]|$)', path.lower())
        file_ext = m.group(1) if m else ''

        if file_ext in _ALLOWED_SCAN_EXTS:
            size = int(head.headers.get('content-length', 0))
            return jsonify({
                'type': 'single',
                'url': url,
                'filename': _unquote(path.split('/')[-1]) or 'document',
                'size_bytes': size,
                'ext': file_ext,
            })

        if 'text/html' not in content_type:
            size = int(head.headers.get('content-length', 0))
            fname = _unquote(path.split('/')[-1]) or 'document'
            return jsonify({'type': 'single', 'url': url,
                            'filename': fname, 'size_bytes': size, 'ext': file_ext})

        try:
            resp = _req.get(url, auth=session_auth, headers=req_headers,
                            allow_redirects=True, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            return jsonify({'error': f'Failed to fetch page: {e}'}), 400

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, 'html.parser')
        base_url = resp.url

        files = []
        seen = set()
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            if not href or href.startswith('#') or href.startswith('?'):
                continue
            full_url = urljoin(base_url, href)
            link_path = _urlparse(full_url).path
            lm = _re.search(r'(\.[a-z0-9]{2,6})(?:[?#]|$)', link_path.lower())
            lext = lm.group(1) if lm else ''
            if lext not in _ALLOWED_SCAN_EXTS:
                continue
            if full_url in seen:
                continue
            seen.add(full_url)
            filename = _unquote(link_path.split('/')[-1]) or 'document'
            size = 0
            try:
                fh = _req.head(full_url, auth=session_auth, headers=req_headers,
                               allow_redirects=True, timeout=4)
                size = int(fh.headers.get('content-length', 0))
            except Exception:
                pass
            files.append({'filename': filename, 'url': full_url,
                          'size_bytes': size, 'ext': lext})

        if not files:
            return jsonify({
                'error': 'No compatible files found at this URL. '
                         'Supported types: PDF, images, Word/Excel/ODT, TXT, EML and more.'
            }), 404

        logger.info(f"scan-url: found {len(files)} compatible files at {url[:80]}")
        return jsonify({'type': 'directory', 'base_url': base_url, 'files': files})

    except Exception as e:
        logger.error(f"scan-url error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@bp.route('/api/upload/history', methods=['GET'])
@login_required
def api_upload_history():
    """Return the last 20 import history rows for the current user."""
    try:
        rows = get_import_history(current_user.id)
        return jsonify({'history': [dict(r) for r in rows]})
    except Exception as e:
        logger.error(f"Failed to fetch upload history: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/upload/from-url', methods=['POST'])
@login_required
def api_upload_from_url():
    """Download file from URL and upload to Paperless."""
    try:
        data = safe_json_body()
        url = data.get('url')
        source = data.get('source', 'url')
        project_slug = data.get('project_slug')
        metadata_in = data.get('metadata', {})
        auth_type = data.get('auth_type', 'none')
        username = data.get('username')
        password = data.get('password')
        token = data.get('token')
        custom_headers = data.get('custom_headers', {})

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # URL-based dedup
        try:
            from analyzer.db import _get_conn as _db_conn_dedup
            with _db_conn_dedup() as _dc:
                _existing = _dc.execute(
                    "SELECT id, filename, paperless_doc_id FROM import_history"
                    " WHERE original_url = ? AND status = 'uploaded' LIMIT 1",
                    (url,)
                ).fetchone()
            if _existing:
                logger.info(f"URL dedup: skipping already-imported URL {url[:80]}")
                return jsonify({
                    'success': True,
                    'duplicate': True,
                    'document_id': _existing['paperless_doc_id'],
                    'title': _existing['filename'],
                    'detail': 'Already imported from this URL',
                }), 200
        except Exception as _dedup_err:
            logger.warning(f"URL dedup check failed (non-fatal): {_dedup_err}")

        from analyzer.remote_downloader import RemoteFileDownloader
        downloader = RemoteFileDownloader()
        try:
            file_path, download_metadata = downloader.download_from_url(
                url=url,
                auth_type=auth_type,
                username=username,
                password=password,
                token=token,
                custom_headers=custom_headers
            )
        except Exception as e:
            log_import(current_user.id, source, url, url=url,
                       status='error', error=f'Download failed: {str(e)}')
            return jsonify({'error': f'Download failed: {str(e)}'}), 400

        filename = download_metadata.get('filename', 'document')

        try:
            if project_slug and metadata_in and current_app.smart_uploader:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        current_app.smart_uploader.upload_to_paperless(file_path, project_slug, metadata_in)
                    )
                finally:
                    loop.close()
                title = metadata_in.get('suggested_title') or filename
            else:
                tag_ids = []
                if project_slug and current_app.paperless_client:
                    proj_tag = current_app.paperless_client.get_or_create_tag(f"project:{project_slug}")
                    if proj_tag is not None:
                        tag_ids.append(proj_tag)
                result = current_app.paperless_client.upload_document(file_path, title=filename,
                                                              tags=tag_ids or None)
                title = filename
        finally:
            downloader.cleanup(file_path)

        if result:
            doc_id = result.get('id')
            if project_slug and not metadata_in and current_app.project_manager:
                current_app.project_manager.increment_document_count(project_slug, delta=1)
            logger.info(f"Uploaded from URL ({source}): {filename}")
            log_import(current_user.id, source, title, url=url,
                       doc_id=doc_id, status='uploaded')
            return jsonify({'success': True, 'document_id': doc_id, 'title': title})
        else:
            # v3.9.4: Paperless returned no result → upstream failure (502) not internal (500)
            log_import(current_user.id, source, filename, url=url,
                       status='error', error='Upload to Paperless failed')
            return jsonify({
                'error': 'Upload to Paperless failed',
                'detail': 'Paperless-ngx rejected the upload (check Paperless logs, API token, and container health)',
                'source': 'paperless-upstream',
            }), 502

    except Exception as e:
        logger.error(f"Failed to upload from URL: {e}")
        try:
            _lc = locals()
            log_import(current_user.id,
                       _lc.get('source', 'url'),
                       _lc.get('url', 'unknown'),
                       url=_lc.get('url'),
                       status='error', error=str(e))
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500


@bp.route('/api/upload/analyze', methods=['POST'])
@login_required
def api_analyze_upload():
    """Analyze uploaded file and extract metadata."""
    if not current_app.smart_uploader:
        return jsonify({'error': 'Smart upload not available (LLM disabled)'}), 503

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        import tempfile
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            metadata = loop.run_until_complete(current_app.smart_uploader.analyze_document(file_path))
        finally:
            loop.close()

        os.remove(file_path)
        os.rmdir(temp_dir)

        logger.info(f"Analyzed upload: {file.filename}")
        return jsonify(metadata)

    except Exception as e:
        logger.error(f"Failed to analyze upload: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/api/upload/submit', methods=['POST'])
@login_required
def api_submit_upload():
    """Submit file to Paperless. Supports direct upload and smart-metadata upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        project_slug = request.form.get('project_slug')
        import json as _json
        metadata_json = request.form.get('metadata', '{}')
        try:
            metadata = _json.loads(metadata_json)
        except (ValueError, TypeError):
            return jsonify({'error': f'metadata must be valid JSON: {metadata_json[:60]!r}'}), 400

        import tempfile
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # v3.9.4: reject zero-byte uploads with clean 400 (was 500 from upstream)
        if os.path.getsize(file_path) == 0:
            try:
                os.remove(file_path)
                os.rmdir(temp_dir)
            except Exception:
                pass
            return jsonify({'error': 'File is empty (0 bytes)'}), 400

        try:
            if project_slug and metadata and current_app.smart_uploader:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        current_app.smart_uploader.upload_to_paperless(file_path, project_slug, metadata)
                    )
                finally:
                    loop.close()
                title = metadata.get('suggested_title') or file.filename
            else:
                tag_ids = []
                if project_slug and current_app.paperless_client:
                    proj_tag = current_app.paperless_client.get_or_create_tag(f"project:{project_slug}")
                    if proj_tag is not None:
                        tag_ids.append(proj_tag)
                result = current_app.paperless_client.upload_document(
                    file_path, title=file.filename.rsplit('.', 1)[0],
                    tags=tag_ids or None
                )
                title = file.filename
        finally:
            os.remove(file_path)
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass

        if result:
            doc_id = result.get('id')
            if project_slug and not metadata and current_app.project_manager:
                current_app.project_manager.increment_document_count(project_slug, delta=1)
            logger.info(f"Uploaded {file.filename} (project={project_slug or 'none'})")
            log_import(current_user.id, 'file', title,
                       doc_id=doc_id, status='uploaded')
            return jsonify({'success': True, 'document_id': doc_id, 'title': title})
        else:
            log_import(current_user.id, 'file', file.filename,
                       status='error', error='Upload to Paperless failed')
            return jsonify({'error': 'Upload failed'}), 500

    except Exception as e:
        logger.error(f"Failed to submit upload: {e}")
        try:
            fname = request.files.get('file', None)
            log_import(current_user.id, 'file',
                       fname.filename if fname else 'unknown',
                       status='error', error=str(e))
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500
