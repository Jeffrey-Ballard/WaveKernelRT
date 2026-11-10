from typing import Generic, Iterator, Protocol, TypeVar
from collections.abc import Iterable

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")

class Source(Protocol[TOutput]):
    def __iter__(self) -> Iterator[TOutput]:
        ...

    def close(self) -> None:
        ...

class Stage(Protocol, Generic[TInput, TOutput]):
    def process(self, block: TInput) -> Iterable[TOutput]:
        ...

class Sink(Protocol[TInput]):
    def write(self, block: TInput) -> None:
        ...

    def close(self) -> None:
        ...
        