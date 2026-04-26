#!/bin/bash
# run_tests.sh
# Standardized execution runner for the ReverieCore memory plugin test suite.
# Automatically includes coverage reporting for both HTML and the terminal.

# Determine directory paths to prevent path-related issues.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
VENV_PYTHON="/home/tom/.hermes/hermes-agent/venv/bin/python"
VENV_PYTEST="/home/tom/.hermes/hermes-agent/venv/bin/pytest"

cd "$SCRIPT_DIR"

echo "================================================="
echo "Running ReverieCore Integration & Unit Tests"
echo "================================================="

# Ensure pytest-cov and anyio/asyncio plugins are present. This prevents CI failures.
$VENV_PYTHON -m pip install -q pytest-cov pytest-asyncio anyio \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-grpc \
    opentelemetry-exporter-otlp-proto-http \
    opentelemetry-semantic-conventions \
    opentelemetry-instrumentation-logging \
    pytest-mock

# Run the pytest suite against the tests directory
$VENV_PYTEST --cov=. --cov-report=term-missing --cov-report=html tests/

status=$?
echo "================================================="
if [ $status -eq 0 ]; then
    echo "✅ Test Suite Passed. View the full HTML coverage report at: htmlcov/index.html"
else
    echo "❌ Test Suite Failed ($status). See above errors."
fi
exit $status
