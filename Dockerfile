FROM python:3.12-slim-bookworm

ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir .
RUN python -m pip check

COPY examples ./examples
COPY tests ./tests
COPY tools ./tools

CMD ["python", "examples/basic_geometries_and_beampatterns/script_example_line_array_geometry_and_beampattern.py", "--output-dir", "/results"]
