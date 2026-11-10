from typing import Generic,TypeVar
from collections.abc import Iterator

from wavekernelrt.pipeline.interfaces import Sink

# keep it generic so we can use this to tee out anything
TBlock = TypeVar("TBlock")

class TeeStage(Generic[TBlock]):
    def __init__(self,sink: Sink[TBlock]) -> None:
        self.sink = sink

    def process(self,block: TBlock) -> Iterator[TBlock]:
        self.sink.write(block)
        yield block

    def close(self) -> None:
        # close it on cleanup
        self.sink.close()
