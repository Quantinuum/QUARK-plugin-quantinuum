def counter_key_to_string_key(counter_key: tuple[int, ...]) -> str:
    return "".join([f"{i}" for i in counter_key])
