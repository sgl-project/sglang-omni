# SPDX-License-Identifier: Apache-2.0
"""Match structured calls against the accepted argument values in BFCL audio rows."""

from collections.abc import Mapping

from pydantic import JsonValue
from scipy.optimize import linear_sum_assignment

ReferenceCall = dict[str, dict[str, list[JsonValue]]]
PredictedCall = tuple[str, dict[str, JsonValue]]


def matches_value(predicted: JsonValue, reference: JsonValue) -> bool:
    if isinstance(reference, dict) and isinstance(predicted, dict):
        return matches_arguments(predicted, reference)
    elif isinstance(reference, list) and isinstance(predicted, list):
        return len(reference) == len(predicted) and all(
            matches_value(actual, expected)
            for actual, expected in zip(predicted, reference)
        )
    else:
        return predicted == reference and isinstance(predicted, bool) == isinstance(
            reference, bool
        )


def matches_arguments(
    arguments: dict[str, JsonValue], reference: Mapping[str, JsonValue]
) -> bool:
    if arguments.keys() - reference.keys():
        return False
    else:
        for parameter, accepted in reference.items():
            alternatives = accepted if isinstance(accepted, list) else [accepted]
            # note (wirybeaver): An empty string among the accepted values permits omitting this argument.
            if parameter in arguments:
                matched = any(
                    matches_value(arguments[parameter], value)
                    for value in alternatives
                    if value != ""
                )
            else:
                matched = "" in alternatives
            if not matched:
                return False
            else:
                pass
        return True


def matches_calls(
    predictions: list[PredictedCall],
    references: list[ReferenceCall],
    required_parameters: dict[str, set[str]],
) -> bool:
    if len(predictions) != len(references):
        return False
    elif not predictions:
        return True
    else:
        matches = [
            [
                name in expected
                and required_parameters[name].issubset(arguments)
                and matches_arguments(arguments, expected[name])
                for name, arguments in predictions
            ]
            for expected in references
        ]
        # note (wirybeaver): Overlapping alternatives can match multiple calls, but each prediction can satisfy only one reference.
        rows, columns = linear_sum_assignment(matches, maximize=True)
        return all(matches[row][column] for row, column in zip(rows, columns))
