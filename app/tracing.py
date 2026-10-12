import time
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.config import settings

logger = logging.getLogger(__name__)

# Configure OpenTelemetry
provider = TracerProvider()
if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        otlp_exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    except Exception as e:
        logger.warning(f"Could not initialize OTLP exporter: {e}")

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("pravah-eval-worker", "0.1.0")

# Langfuse client setup
langfuse_client = None
if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
    try:
        from langfuse import Langfuse
        langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST
        )
        logger.info("Langfuse observability initialized successfully.")
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse client: {e}")


@contextmanager
def trace_eval_span(name: str, input_data: Any, metadata: Optional[Dict[str, Any]] = None):
    """
    Context manager that records latency, tokens, cost, and metadata to OpenTelemetry and Langfuse.
    """
    start_time = time.time()
    span_metadata = metadata or {}
    record = {
        "name": name,
        "input": input_data,
        "output": None,
        "latency_ms": 0.0,
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
        "cost_usd": 0.0,
        "metadata": span_metadata
    }

    with tracer.start_as_current_span(name) as otel_span:
        otel_span.set_attribute("eval.name", name)
        for k, v in span_metadata.items():
            if isinstance(v, (str, int, float, bool)):
                otel_span.set_attribute(f"eval.meta.{k}", v)

        try:
            yield record
        finally:
            latency = (time.time() - start_time) * 1000
            record["latency_ms"] = round(latency, 2)
            otel_span.set_attribute("eval.latency_ms", record["latency_ms"])
            otel_span.set_attribute("eval.prompt_tokens", record["tokens"]["prompt"])
            otel_span.set_attribute("eval.completion_tokens", record["tokens"]["completion"])
            otel_span.set_attribute("eval.cost_usd", record["cost_usd"])

            # Export to Langfuse if available
            if langfuse_client:
                try:
                    langfuse_client.trace(
                        name=name,
                        input=input_data,
                        output=record.get("output"),
                        metadata={
                            **span_metadata,
                            "latency_ms": record["latency_ms"],
                            "tokens": record["tokens"],
                            "cost_usd": record["cost_usd"]
                        }
                    )
                except Exception as ex:
                    logger.debug(f"Langfuse trace logging skipped: {ex}")
