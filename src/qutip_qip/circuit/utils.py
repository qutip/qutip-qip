from typing import Iterable


def _check_iterable(input_name: str, input_value: any):
    try:
        iter(input_value)
    except TypeError:
        raise TypeError(
            f"{input_name} must be an iterable input, got {input_value}."
        )


def _check_limit_(
    input_name: str, input_value: Iterable, limit, element_type: type = int
):
    for e in input_value:
        if type(e) is not element_type:
            raise TypeError(
                f"Each entry of {input_name} must be less than {limit}, got {input_value}."
            )

        if e > limit:
            raise ValueError(
                f"Each entry of {input_name} must be less than {limit}, got {input_value}."
            )
