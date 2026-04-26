import pytest
import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from reveriecore.enrichment import EnrichmentService, EnrichmentHandler, EnrichmentContext
from reveriecore.retrieval import Retriever, RetrievalHandler, RetrievalContext
from reveriecore.telemetry import initialize_telemetry, get_tracer

@pytest.fixture(scope="module")
def otel_setup():
    # 1. Ensure telemetry is initialized
    initialize_telemetry()
    
    # 2. Get the active provider
    provider = trace.get_tracer_provider()
    
    # 3. Attach InMemorySpanExporter via SimpleSpanProcessor
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    
    # If it's a ProxyTracerProvider, we can't add processors.
    # But initialize_telemetry should have set it to a real TracerProvider.
    if hasattr(provider, "add_span_processor"):
        provider.add_span_processor(processor)
    else:
        # Fallback: if it's still a proxy, force a real one (though this might still fail global calls)
        from opentelemetry.sdk.trace import TracerProvider
        provider = TracerProvider()
        provider.add_span_processor(processor)
        try:
            trace.set_tracer_provider(provider)
        except Exception:
            pass # Already set
            
    return exporter

def test_enrichment_telemetry(otel_setup):
    exporter = otel_setup
    exporter.clear()
    
    service = EnrichmentService()
    # Mock handlers to add some delay for duration check
    class SlowHandler(EnrichmentHandler):
        def process(self, context, service):
            time.sleep(0.01)
            context.importance_score = 7.0
            
    service.analysis_pipeline = [SlowHandler()]
    
    service.enrich("Test telemetry ingestion")
    
    spans = exporter.get_finished_spans()
    assert len(spans) >= 2 # Parent + 1 handler
    
    parent = next(s for s in spans if s.name == "reverie.enrichment")
    handler_span = next(s for s in spans if "SlowHandler" in s.name)
    
    assert parent.status.status_code == StatusCode.UNSET
    assert handler_span.parent.span_id == parent.context.span_id
    
    # Trace Latency Sanity Check
    duration_ns = handler_span.end_time - handler_span.start_time
    assert duration_ns > 0, "Span duration should be calculated and greater than 0"

def test_exception_recording(otel_setup):
    exporter = otel_setup
    exporter.clear()
    
    service = EnrichmentService()
    class FailingHandler(EnrichmentHandler):
        def process(self, context, service):
            raise ValueError("Simulated Failure")
            
    service.analysis_pipeline = [FailingHandler()]
    
    # This shouldn't raise since we catch in orchestrator
    service.enrich("Test failure ingestion")
    
    spans = exporter.get_finished_spans()
    handler_span = next(s for s in spans if "FailingHandler" in s.name)
    
    assert handler_span.status.status_code == StatusCode.ERROR
    assert len(handler_span.events) > 0
    # OTel records type and message separately
    assert handler_span.events[0].attributes["exception.message"] == "Simulated Failure"

def test_retrieval_telemetry(otel_setup, mocker):
    exporter = otel_setup
    exporter.clear()
    
    db = mocker.Mock()
    retriever = Retriever(db)
    
    class MockDiscovery(RetrievalHandler):
        def process(self, context, retriever):
            context.candidates[1] = {"id": 1, "importance": 5.0, "learned_at": "2026-01-01T00:00:00", "source": "test"}
            
    retriever.discovery_pipeline = [MockDiscovery()]
    retriever.ranking_pipeline = []
    retriever.budget_pipeline = []
    
    retriever.search([0.1]*384, query_text="test", limit=5)
    
    spans = exporter.get_finished_spans()
    parent = next(s for s in spans if s.name == "reverie.retrieval")
    handler_span = next(s for s in spans if "MockDiscovery" in s.name)
    
    assert parent.attributes["retrieval.result_count"] == 0 # we didn't add to results
    assert handler_span.attributes["retrieval.candidate_count"] == 1
