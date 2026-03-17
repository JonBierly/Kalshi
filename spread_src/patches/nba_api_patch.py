"""
Patch for nba_api live endpoints to bypass CDN blocking.
cdn.nba.com frequently blocks datacenter/VPS IPs, returning HTML 403 pages
instead of JSON. This patch transparently routes live CDN requests to the
unblocked AWS S3 bucket that serves the same data.
"""
import logging
import requests
import requests.sessions

logger = logging.getLogger(__name__)

CDN_PATH = 'cdn.nba.com/static/json/liveData'
S3_PATH  = 'nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData'

# Save original methods ONCE at import time
_original_get = requests.get
_original_session_get = requests.sessions.Session.get


def _rewrite_url(url):
    """Rewrite cdn.nba.com live URLs to the S3 bucket."""
    if CDN_PATH in url:
        return url.replace('cdn.nba.com/static/json/liveData', S3_PATH)
    return url


def _clean_headers_for_s3(headers):
    """
    Strip headers that break S3 requests.
    nba_api hardcodes 'Host: cdn.nba.com' in its STATS_HEADERS dict.
    Sending that Host header to S3 causes a 403 Forbidden.
    """
    if not headers:
        return headers
    cleaned = {k: v for k, v in headers.items()
               if k.lower() not in ('host',)}
    cleaned.setdefault('User-Agent', 'Mozilla/5.0')
    return cleaned


def _patched_get(url, *args, **kwargs):
    new_url = _rewrite_url(url)
    if new_url != url:
        url = new_url
        kwargs['headers'] = _clean_headers_for_s3(kwargs.get('headers'))
        kwargs.setdefault('timeout', 15)
    return _original_get(url, *args, **kwargs)


def _patched_session_get(self, url, **kwargs):
    new_url = _rewrite_url(url)
    if new_url != url:
        url = new_url
        # Clean both per-request headers AND session-level headers
        kwargs['headers'] = _clean_headers_for_s3(kwargs.get('headers'))
        self.headers = _clean_headers_for_s3(dict(self.headers))
        kwargs.setdefault('timeout', 15)
    return _original_session_get(self, url, **kwargs)


def apply_patch():
    """Apply monkeypatch to reroute nba_api CDN calls to S3."""
    try:
        requests.get = _patched_get
        requests.Session.get = _patched_session_get
        requests.sessions.Session.get = _patched_session_get
        logger.info("Patched requests for nba_api CDN→S3 bypass")
    except Exception as e:
        logger.error(f"Failed to apply nba_api patch: {e}")
