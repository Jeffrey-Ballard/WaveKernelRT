from dataclasses import dataclass
from numbers import Integral, Real
from math import isfinite
from datetime import datetime, timezone

@dataclass(frozen=True, slots=True)
class Timebase:
    """
    Mapping between global sample indices and UTC time.

    origin_sample:
        Global sample index corresponding to origin_unix_ns.

    origin_unix_ns:
        UTC Unix timestamp of origin_sample, in nanoseconds.
    """

    origin_sample: int| float
    origin_unix_ns: int
    sample_rate_hz: float

    def sample_to_unix_ns(self, sample_index: int | float) -> int:
        if isinstance(sample_index, Integral):
            sample_index = int(sample_index)

        sample_offset = sample_index - self.origin_sample

        time_offset_ns = round(sample_offset * 1_000_000_000 / self.sample_rate_hz)

        return self.origin_unix_ns + time_offset_ns

    def unix_ns_to_sample(self, unix_ns: int ) -> int | float:
        if isinstance(unix_ns, bool) or not isinstance(unix_ns, Integral):
            raise ValueError("unix_ns must be an integer nanosecond timestamp")

        time_offset_ns = int(unix_ns) - self.origin_unix_ns
        sample_offset = time_offset_ns * self.sample_rate_hz / 1_000_000_000

        return self.origin_sample + sample_offset

    def sample_to_datetime(self, sample_index: int) -> datetime:
        unix_ns = self.sample_to_unix_ns(sample_index)

        return datetime.fromtimestamp( unix_ns / 1_000_000_000, tz=timezone.utc)

    def __post_init__(self) -> None:

        if (isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, Real)
            or not isfinite(self.sample_rate_hz)
            or self.sample_rate_hz <= 0):
            raise ValueError("sample_rate_hz must be a finite positive number")

        if (isinstance(self.origin_unix_ns, bool)
            or not isinstance(self.origin_unix_ns, Integral)):
            raise ValueError("origin_unix_ns must be an integer nanosecond timestamp")

        # Normalize NumPy integers to Python int for timestamp arithmetic.
        object.__setattr__(self, "origin_unix_ns", int(self.origin_unix_ns))

        if isinstance(self.origin_sample, Integral):
            object.__setattr__(self, "origin_sample", int(self.origin_sample))