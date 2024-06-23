class ConstantSchedule(object):
    def __init__(self, value):
        self._v = value

    def value(self, t):
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, waypoints, interpolation=linear_interpolation, outside_value=None):
        """
        :param waypoints: [(int, int)] list of pairs '(time, value)' meanining that schedule
        should output 'value' when 't==time'. All the times must be sorted in an increasing order. 
        :param interpolation: lambda float, float, float: float
        a function that takes in the two closest waypoint values at time t and the fraction of
        distance from left waypoint to right waypoint that t has covered. 
        :param outside_value: returned value for t not between two waypoints

        :author: CS294 UC Berkeley
        """
        idxes = [e[0] for e in waypoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._waypoints = waypoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._waypoints[:-1], self._waypoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps (afterwards final_p is returned).

        :author: CS294 UC Berkeley
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)