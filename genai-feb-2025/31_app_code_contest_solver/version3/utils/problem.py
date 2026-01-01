from dataclasses import dataclass


@dataclass
class Problem:
    src_uid: str
    description: str
    input_spec: str
    output_spec: str
    sample_inputs: list[str]
    sample_outputs: list[str]
    notes: str
    unit_tests: list[dict]
    input_from: str = "standard input"
    output_to: str = "standard output"
