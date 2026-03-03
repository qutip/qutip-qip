from typing import TypeVar, Sequence
import numpy as np
from qutip_qip.typing import Int, IntSequence

T = TypeVar("T")  # T can be any type


def check_limit(
    input_name: str, input_value: Sequence[T], lower_limit: T, upper_limit: T
):
    if len(input_value) == 0:
        return

    min_element = min(input_value)
    if min_element < lower_limit:
        raise ValueError(
            f"Each entry of {input_name} must be greater than {lower_limit}, but found {min_element}."
        )

    max_element = max(input_value)
    if max_element > upper_limit:
        raise ValueError(
            f"Each entry of {input_name} must be less than {upper_limit}, but found {max_element}."
        )


def convert_int_to_list(
    input_name: str, input_value: Int | IntSequence
) -> IntSequence:
    if isinstance(input_value, Int):
        return [input_value]

    elif isinstance(input_value, Sequence) and not isinstance(
        input_value, str
    ):
        for i, val in enumerate(input_value):
            if not isinstance(val, Int):
                raise TypeError(
                    f"All elements in '{input_name}' must be integers. "
                    f"Found {type(val).__name__} ({val}) at index {i}."
                )
        return input_value
    else:
        raise TypeError(
            f"{input_name} must be an int or sequence of int, got {input_value}."
        )
