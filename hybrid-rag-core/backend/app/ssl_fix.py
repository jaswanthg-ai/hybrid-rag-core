"""Global SSL fix for corporate networks. Import once at startup."""

import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_orig_init = urllib3.HTTPSConnectionPool.__init__


def _patched_init(self, *args, **kwargs):
    kwargs["cert_reqs"] = "CERT_NONE"
    _orig_init(self, *args, **kwargs)


urllib3.HTTPSConnectionPool.__init__ = _patched_init
