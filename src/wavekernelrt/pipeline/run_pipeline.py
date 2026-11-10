"""Synchronous execution of WaveKernelRT source-stage-sink pipelines."""

from collections.abc import Iterable
from contextlib import ExitStack

from wavekernelrt.pipeline.interfaces import Source, Stage, Sink

def run_pipeline(source: Source, stages: list[Stage], sink: Sink) -> None:
     """Run a source through ordered stages and write every final block.

     Parameters
     ----------
     source : Source
         Iterable yielding runtime data blocks and providing ``close()``.
     stages : list of Stage
         Ordered processors. Each ``process(block)`` call yields zero or more
         output blocks; a stage may optionally provide ``close()``.
     sink : Sink
         Destination providing ``write(block)`` and ``close()``.

     Returns
     -------
     None

     Notes
     -----
     Execution is synchronous and single-process. Blocks flow lazily through
     every stage before another source block is requested. There is no
     end-of-stream flush protocol, so incomplete buffered groups are discarded.

     Closeable stages, the sink, and the source are closed even if iteration,
     processing, or writing raises. Cleanup errors follow normal
     ``contextlib.ExitStack`` behavior.
     """
     with ExitStack() as cleanup:
        # Callbacks run in reverse registration order:
        # stages, then sink, then source.
        cleanup.callback(source.close)
        cleanup.callback(sink.close)

        for stage in reversed(stages):
            close = getattr(stage, "close", None)
            if callable(close):
                cleanup.callback(close)

        for source_block in source:
            blocks = (source_block,)

            for stage in stages:
                blocks = _process_stage(stage, blocks)

            for output_block in blocks:
                sink.write(output_block)

def _process_stage(stage: Stage, input_blocks: Iterable) -> Iterable:
    # Pass every input block through one stage
    for input_block in input_blocks:
        yield from stage.process(input_block)
