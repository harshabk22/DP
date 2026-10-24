"""Global pytest configuration for MessMate backend tests."""

import os

# Guarantee test environment flag for all test runs
os.environ["ENVIRONMENT"] = "test"
os.environ["SECRET_KEY"] = "messmate-production-secure-hmac-sha256-key-v1-super-secret-512bit"
