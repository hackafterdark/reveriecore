import logging
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
except ImportError:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        OTLPSpanExporter = None

from opentelemetry.instrumentation.logging import LoggingInstrumentor
from urllib.parse import urlparse
import socket

logger = logging.getLogger(__name__)

def _is_endpoint_reachable(endpoint: str) -> bool:
    """Checks if the telemetry endpoint is reachable via TCP."""
    try:
        # Default OTel ports
        default_port = 4318 if "http" in endpoint.lower() else 4317
        
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or default_port
        
        # If the endpoint was just "localhost:4317" (no scheme), urlparse might fail to get hostname
        if not parsed.scheme and ":" in endpoint:
            parts = endpoint.split(":")
            host = parts[0]
            port = int(parts[1])

        with socket.create_connection((host, port), timeout=1.0):
            return True
    except Exception:
        return False


_is_initialized = False

def initialize_telemetry(service_name: str = "reveriecore", endpoint: str = None, enabled: bool = True):
    """Initializes OpenTelemetry Tracer Provider and Exporters."""
    global _is_initialized
    if _is_initialized:
        return
    
    if not enabled:
        logger.info("Telemetry is explicitly disabled via configuration.")
        _is_initialized = True
        return

    # 1. Setup Resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
    })

    # 2. Setup Tracer Provider
    provider = TracerProvider(resource=resource)
    
    # 3. Setup OTLP Exporter
    # If no endpoint is provided, we check defaults based on what's installed
    check_endpoint = endpoint
    if not check_endpoint:
        # Determine default based on what exporter class we have
        # OTLPSpanExporter is imported at the top of the file
        is_grpc = "grpc" in str(OTLPSpanExporter) if OTLPSpanExporter else False
        check_endpoint = "http://localhost:4317" if is_grpc else "http://localhost:4318"

    try:
        if OTLPSpanExporter:
            if _is_endpoint_reachable(check_endpoint):
                otlp_exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
                processor = BatchSpanProcessor(otlp_exporter)
                provider.add_span_processor(processor)
                logger.info(f"OpenTelemetry initialized with OTLP Exporter at {check_endpoint}")
            else:
                logger.warning(f"Telemetry endpoint {check_endpoint} not reachable. Exporting disabled to prevent noise.")
        else:
            logger.warning("OTLPSpanExporter not available. Spans will not be exported.")
    except Exception as e:
        logger.warning(f"Failed to initialize OTLP Exporter: {e}. Falling back to No-Op.")

    # 4. Set Global Tracer Provider
    trace.set_tracer_provider(provider)
    
    # 5. Initialize Logging Instrumentation
    LoggingInstrumentor().instrument(set_logging_format=True)
    
    _is_initialized = True


def get_tracer(name: str = "reveriecore"):
    """Returns a tracer instance."""
    return trace.get_tracer(name)
