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

logger = logging.getLogger(__name__)

_is_initialized = False

def initialize_telemetry(service_name: str = "reveriecore"):
    """Initializes OpenTelemetry Tracer Provider and Exporters."""
    global _is_initialized
    if _is_initialized:
        return
    
    # 1. Setup Resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
    })

    # 2. Setup Tracer Provider
    provider = TracerProvider(resource=resource)
    
    # 3. Setup OTLP Exporter (Defaults to localhost:4317 or 4318)
    try:
        if OTLPSpanExporter:
            otlp_exporter = OTLPSpanExporter()
            processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(processor)
        else:
            logger.warning("OTLPSpanExporter not available. Spans will not be exported.")
    except Exception as e:
        logger.warning(f"Failed to initialize OTLP Exporter: {e}. Falling back to No-Op.")

    # 4. Set Global Tracer Provider
    trace.set_tracer_provider(provider)
    
    # 5. Initialize Logging Instrumentation
    # This injects [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] into log records
    LoggingInstrumentor().instrument(set_logging_format=True)
    
    _is_initialized = True
    logger.info("OpenTelemetry initialized with OTLP Exporter and Logging Instrumentation.")

def get_tracer(name: str = "reveriecore"):
    """Returns a tracer instance."""
    return trace.get_tracer(name)
