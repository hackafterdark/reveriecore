import logging
import os
from typing import Dict, Any, Optional, Union
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
from importlib.metadata import version, metadata, PackageNotFoundError



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
        
def get_service_info(package_name: str = "reveriecore") -> Dict[str, str]:
    """Dynamically retrieves the name and version from package metadata."""
    try:
        dist_metadata = metadata(package_name)
        return {
            "name": dist_metadata.get("Name", package_name),
            "version": dist_metadata.get("Version", "0.0.0-dev")
        }
    except PackageNotFoundError:
        return {
            "name": package_name,
            "version": "0.0.0-dev"
        }




_is_initialized = False

def initialize_telemetry(
    service_name: str = None, 
    endpoint: str = None, 
    headers: Union[str, Dict[str, str]] = None,
    protocol: str = None,
    resource_attributes: Dict[str, Any] = None,
    enabled: bool = True
):

    """Initializes OpenTelemetry Tracer Provider and Exporters."""
    global _is_initialized
    if _is_initialized:
        return
    
    if not enabled:
        logger.info("Telemetry is explicitly disabled via configuration.")
        _is_initialized = True
        return

    # 1. Setup Resource
    # Combine default attributes with user-provided ones
    info = get_service_info(service_name or "reveriecore")
    base_attributes = {
        "service.name": info["name"],
        "service.version": info["version"],
    }


    if resource_attributes:
        base_attributes.update(resource_attributes)
        
    resource = Resource.create(base_attributes)

    # 2. Setup Tracer Provider
    provider = TracerProvider(resource=resource)
    
    # 3. Setup OTLP Exporter
    # Parse headers if provided as string "key=val,key2=val2"
    final_headers = headers
    if isinstance(headers, str):
        final_headers = {}
        for pair in headers.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                final_headers[k.strip()] = v.strip()

    check_endpoint = endpoint
    if not check_endpoint:
        is_grpc = "grpc" in str(OTLPSpanExporter) if OTLPSpanExporter else False
        check_endpoint = "http://localhost:4317" if is_grpc else "http://localhost:4318"

    try:
        if OTLPSpanExporter:
            if _is_endpoint_reachable(check_endpoint):
                exporter_kwargs = {}
                if endpoint:
                    exporter_kwargs["endpoint"] = endpoint
                if final_headers:
                    exporter_kwargs["headers"] = final_headers
                if protocol:
                    exporter_kwargs["protocol"] = protocol
                
                otlp_exporter = OTLPSpanExporter(**exporter_kwargs)
                processor = BatchSpanProcessor(otlp_exporter)
                provider.add_span_processor(processor)
                logger.info(f"OpenTelemetry initialized with OTLP Exporter at {check_endpoint} (Protocol: {protocol or 'default'})")
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
