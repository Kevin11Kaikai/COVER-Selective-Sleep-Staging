"""Display names only: never used in frozen keys or policy tie hashes."""


def method_name(value: str) -> str:
    return "COVER" if value == "Learned-Selected" else value
