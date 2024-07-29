
from typing import List, Tuple, Callable, Optional

from core.schedules import Schedule
from core.utils import linear_interpolation


class PiecewiseSchedule(Schedule):
    def __init__(self,
                 endpoints: List[Tuple[int, float]],
                 interpolation: Callable[[float, float, float], float] = linear_interpolation,
                 outside_value: Optional[float] = None):
        """
        Args:
            endpoints (List[Tuple[int,float]]): A list of tuples
                `(t, value)` such that the output
                is an interpolation (given by the `interpolation` callable)
                between two values.
                E.g.
                t=400 and endpoints=[(0, 20.0),(500, 30.0)]
                output=20.0 + 0.8 * (30.0 - 20.0) = 28.0
                NOTE: All the values for time must be sorted in an increasing
                order.

            interpolation (callable): A function that takes the left-value,
                the right-value and an alpha interpolation parameter
                (0.0=only left value, 1.0=only right value), which is the
                fraction of distance from left endpoint to right endpoint.

            outside_value (Optional[float]): If t in call to `value` is
                outside of all the intervals in `endpoints` this value is
                returned. If None then an AssertionError is raised when outside
                value is requested.
        """
        self.endpoints = endpoints
        self.interpolation = interpolation
        self.outside_value = outside_value

        assert len(self.endpoints) > 1, "endpoints should contain at least two points"
        for i in range(len(self.endpoints) - 1):
            assert self.endpoints[i][0] < self.endpoints[i + 1][0], "endpoints should be sorted in increasing order"

    def value(self, t: int) -> float:
        if t < self.endpoints[0][0] or t > self.endpoints[-1][0]:
            if self.outside_value is not None:
                return self.outside_value
            else:
                raise AssertionError("Value requested is outside the range of the endpoints and no outside_value is set.")

        for (left_t, left_value), (right_t, right_value) in zip(self.endpoints[:-1], self.endpoints[1:]):
            if left_t <= t <= right_t:
                alpha = (t - left_t) / (right_t - left_t)
                return self.interpolation(left_value, right_value, alpha)

        raise AssertionError("Failed to compute the value for the given time.")
