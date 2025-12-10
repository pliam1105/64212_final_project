"""Controllers for differential IK / pseudo-inverse velocity mapping."""

from __future__ import annotations

import numpy as np

from pydrake.all import (
    BasicVector,
    Context,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    Trajectory,
)


class PseudoInverseController(LeafSystem):
    """Maps spatial velocity at the gripper to joint velocities via J^+."""

    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._wsg = plant.GetModelInstanceByName("wsg")
        self._G = plant.GetBodyByName("body", self._wsg).body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context: Context, output: BasicVector) -> None:
        V_G = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )[:, self.iiwa_start : self.iiwa_end + 1]
        v = np.linalg.pinv(J_G) @ V_G
        output.SetFromVector(v)


def make_trajectory(
    X_Gs: list[RigidTransform], finger_values: np.ndarray, sample_times: list[float]
) -> tuple[Trajectory, PiecewisePolynomial]:
    robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, X_Gs)
    robot_velocity_trajectory = robot_position_trajectory.MakeDerivative()
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
    return robot_velocity_trajectory, traj_wsg_command


class CompliantPullCommand(LeafSystem):
    """
    Outputs a spatial velocity V_WG that pulls along the drawer axis in world.
    It keeps pulling until the iiwa joint-torque norm exceeds a threshold.

    Debug: prints when switching between 'moving' and 'stopped'.
    """

    def __init__(self,
                 drawer_axis_W,
                 v_nominal: float = 0.03,   # m/s along drawer
                 tau_stop: float = 500.0):   # Nm threshold on ||tau|| 25
        super().__init__()

        a = np.asarray(drawer_axis_W, dtype=float).reshape(3,)
        self._a = a / np.linalg.norm(a)

        self._v_nominal = v_nominal
        self._tau_stop = tau_stop

        # For debug: remember last "moving" / "stopped" mode
        self._last_moving = None

        # Inputs: iiwa joint positions (unused here but nice for future) and measured torques
        self._q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self._tau_meas_port = self.DeclareVectorInputPort("iiwa.torque_measured", 7)

        # Output: spatial velocity V_WG (6)
        self.DeclareVectorOutputPort("V_WG", 6, self.CalcOutput)

    def CalcOutput(self, context: Context, output: BasicVector) -> None:
        # We only use torques for compliance right now
        tau_meas = self._tau_meas_port.Eval(context)
        effort = float(np.linalg.norm(tau_meas))

        if effort < self._tau_stop:
            v_axis = self._v_nominal
            moving = True
        else:
            v_axis = 0.0
            moving = False

        # --- DEBUG: print when switching modes ---
        if self._last_moving is None or moving != self._last_moving:
            state_str = "MOVING" if moving else "STOPPED"
            print(
                f"[CompliantPull] t={context.get_time():.3f}  "
                f"mode={state_str}  effort={effort:.2f}  "
                f"tau_stop={self._tau_stop:.2f}  v_axis={v_axis:.3f}"
            )
            self._last_moving = moving
        # ----------------------------------------

        V_lin = v_axis * self._a
        V_ang = np.zeros(3)
        V_WG = np.hstack([V_ang, V_lin])
        output.SetFromVector(V_WG)



# --------------------------------------------------------------------------
# Switcher: trajectory phase then compliant phase
# --------------------------------------------------------------------------

class VwgSwitcher(LeafSystem):
    """
    For t < switch_time:  use trajectory V_WG
    For t >= switch_time: use compliant V_WG
    """
    def __init__(self, switch_time: float):
        super().__init__()
        self._switch_time = switch_time

        self._traj_port = self.DeclareVectorInputPort("V_WG_traj", 6)
        self._comp_port = self.DeclareVectorInputPort("V_WG_compliant", 6)
        self.DeclareVectorOutputPort("V_WG", 6, self.CalcOutput)

    def CalcOutput(self, context, output: BasicVector) -> None:
        t = context.get_time()
        if t < self._switch_time:
            V = self._traj_port.Eval(context)
        else:
            
            V = self._comp_port.Eval(context)
            # print("TEST: ", V)
        output.SetFromVector(V)