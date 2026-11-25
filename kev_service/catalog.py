"""Immutable source pins. Change deliberately, rebuild, and run the real smoke tests."""
UPSTREAM_COMMIT = "65b1e763943ace53fd3a4aae00d8d08dbd849772"
MODELS = {
    "kev-0.8b": {"repo": "jaredpalmer/kev-0.8b", "revision": "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"},
    "kev-4b": {"repo": "jaredpalmer/kev-4b", "revision": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"},
    "kev-9b": {"repo": "jaredpalmer/kev-9b", "revision": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"},
    "kev-27b": {"repo": "jaredpalmer/kev-27b", "revision": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf"},
}
ALIASES = ("kev-latest", "jev-latest")

