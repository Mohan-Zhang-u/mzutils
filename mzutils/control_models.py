from typing import Union, List

import numpy as np


class PIDModel(object):
    def __init__(
            self, Kp: Union[None, float, List[float]],
            Ki: Union[None, float, List[float]],
            Kd: Union[None, float, List[float]],
            setpoints: List[float],
            lower_bounds: Union[None, float, List[float]] = None,
            upper_bounds: Union[None, float, List[float]] = None,
            steady_actions: Union[None, List[float]] = None):
        """
        :param Kp: proportional gain
        :param Ki: integral gain
        :param Kd: derivative gain
        :param setpoints: setpoints (steady states) for the PID controller output
        :param lower_bounds: lower bounds for the PID controller output
        :param upper_bounds: upper bounds for the PID controller output
        :param steady_actions: actions to be used when the setpoints are reached, optional. If given, the control will be around the steady actions.
        """
        len_c = len(setpoints)
        self.len_c = len_c
        if isinstance(Kp, float):
            Kp = [Kp for i in range(len_c)]
        if isinstance(Ki, float):
            Ki = [Ki for i in range(len_c)]
        if isinstance(Kd, float):
            Kd = [Kd for i in range(len_c)]
        if steady_actions is None:
            # do not adjust bias
            steady_actions = [0.0 for i in range(len_c)]
        if lower_bounds is None or isinstance(lower_bounds, float):
            lower_bounds = [lower_bounds for i in range(len_c)]
        if isinstance(lower_bounds[0], float):
            lower_bounds = [lower_bounds[i] - steady_actions[i] for i in range(len_c)]
        if upper_bounds is None or isinstance(upper_bounds, float):
            upper_bounds = [upper_bounds for i in range(len_c)]
        if isinstance(upper_bounds[0], float):
            upper_bounds = [upper_bounds[i] - steady_actions[i] for i in range(len_c)]
        from simple_pid import PID
        self.pids = []
        for i in range(len_c):
            self.pids.append(PID(Kp[i], Ki[i], Kd[i], setpoint=setpoints[i], output_limits=(
                lower_bounds[i], upper_bounds[i])))
        self.steady_actions = steady_actions

    def predict(self, state):
        """
        :param state: control state of the system. The state is a list of length len_c, each of the element cooresponding to an action.
        """
        assert len(state) == self.len_c
        actions = []
        for i in range(len(state)):
            self.pids[i](state[i])
            actions.append(self.pids[i](state[i]) + self.steady_actions[i])
        actions = np.array(actions)
        return actions
