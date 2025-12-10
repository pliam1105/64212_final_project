from __future__ import annotations

from typing import Optional

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import Diagram, DiagramBuilder
from pydrake.all import PointCloud

from open_drawer import open_drawer, plan_open_drawer
import numpy as np
from pydrake.all import Integrator, Simulator, PiecewisePolynomial, TrajectorySource, ConstantVectorSource
from controller import PseudoInverseController, CompliantPullCommand, VwgSwitcher


class StateMachine:
    def __init__(
        self,
        meshcat: Optional[object] = None,
        *,
        simulator: Optional[object] = None,
    ) -> None:
        """Build a single persistent Diagram and Simulator for all stages.

        This constructor creates a Diagram that contains the station and
        the controller/trajectory sources (placeholders). It then builds
        a Simulator that is reused across state transitions.
        """
        # Build station without default iiwa/wsg sources so we can add
        # persistent controllers/sources below.
        from scene_setup import build_station_setup

        builder, station, meshcat = build_station_setup(
            meshcat=meshcat, connect_default_iiwa=False, connect_default_wsg=False
        )

        self.meshcat = meshcat
        self.station = station

        # The station_system is already in the builder (added by build_station_setup)
        # We need to retrieve it from the builder
        self.station_system = builder.GetSubsystemByName("station")

        # Plant access
        plant = self.station_system.GetSubsystemByName("plant")

        # Placeholder V_WG source: start with zero spatial velocity
        from pydrake.all import PiecewisePolynomial, TrajectorySource, ConstantVectorSource

        zero_vals = np.zeros((6, 2))
        zero_traj = PiecewisePolynomial.FirstOrderHold([0.0, 1.0], zero_vals)
        self.V_G_source = builder.AddSystem(TrajectorySource(zero_traj))

        # Placeholder WSG trajectory source: keep gripper slightly open by default
        wsg_port = self.station_system.GetInputPort("wsg.position")
        wsg_opening = np.full((1, 2), 0.05)
        wsg_zero_traj = PiecewisePolynomial.FirstOrderHold([0.0, 1.0], wsg_opening)
        self.wsg_source = builder.AddSystem(TrajectorySource(wsg_zero_traj))
        builder.Connect(self.wsg_source.get_output_port(), self.station_system.GetInputPort("wsg.position"))

        # Compliant pull and switcher
        self.pull_cmd = builder.AddSystem(CompliantPullCommand(drawer_axis_W=np.array([0.0, -1.0, 0.0])))
        builder.Connect(self.station_system.GetOutputPort("iiwa.position_measured"), self.pull_cmd.get_input_port(0))
        builder.Connect(self.station_system.GetOutputPort("iiwa.torque_measured"), self.pull_cmd.get_input_port(1))

        # Switcher placeholder (switch time will be set when trajectories are known)
        self.switcher = builder.AddSystem(VwgSwitcher(switch_time=1.0))
        builder.Connect(self.V_G_source.get_output_port(), self.switcher.get_input_port(0))
        builder.Connect(self.pull_cmd.get_output_port(), self.switcher.get_input_port(1))

        # Pseudo-inverse controller + integrator
        self.controller = builder.AddSystem(PseudoInverseController(plant))
        self.integrator = builder.AddSystem(Integrator(7))

        builder.Connect(self.switcher.get_output_port(), self.controller.GetInputPort("V_WG"))
        builder.Connect(self.controller.get_output_port(), self.integrator.get_input_port())
        builder.Connect(self.integrator.get_output_port(), self.station_system.GetInputPort("iiwa.position"))
        builder.Connect(self.station_system.GetOutputPort("iiwa.position_measured"), self.controller.GetInputPort("iiwa.position"))

        # Note: Station is in position-only control mode, so no torque input port exists.
        # The integrator output is connected directly to iiwa.position.

        # iiwa = plant.GetModelInstanceByName("iiwa")
        # nq_act = plant.num_actuated_dofs(iiwa)
        # zero_ff = builder.AddSystem(ConstantVectorSource(np.zeros(nq_act)))
        # builder.Connect(zero_ff.get_output_port(), station.GetInputPort("iiwa.torque"))

        # Point clouds exported by build_station_setup are already wired
        # Build the diagram and simulator once
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.simulator = Simulator(self.diagram, self.context)

        # Initialize integrator to current iiwa positions
        plant_ctx = self.diagram.GetMutableSubsystemContext(plant, self.context)
        q0 = plant.GetPositions(plant_ctx, plant.GetModelInstanceByName("iiwa"))
        integ_ctx = self.integrator.GetMyContextFromRoot(self.context)
        self.integrator.set_integral_value(integ_ctx, q0)

        # initial state and storage
        self.state = "preRANSAC"
        self.camera_point_cloud = None
        self.home_q = None
        return

    def step(self) -> str:
        """Run the actions for the current state and (optionally)
        transition to the next state.

        Returns the name of the new state after stepping.
        """
        if self.state == "preRANSAC":
            return self._enter_preRANSAC()
        elif self.state == "open_drawer":
            return self._enter_open_drawer()
        elif self.state == "home":
            return self._enter_home()
        else:
            return self.state

    def _enter_preRANSAC(self) -> str:
        """Move the arm to the camera pose, publish once, and read the
        camera point cloud. Then transition to `open_drawer`.
        """
        # Import the helper from the perception/motion module. The
        # function `move_wsg_camera_to_initial_pose` performs IK and
        # updates the plant state so that the hand-mounted camera ends
        # up at the desired camera pose.
        try:
            from open_drawer_RANSAC import move_wsg_camera_to_initial_pose
        except Exception:
            raise ImportError(
                "Could not import move_wsg_camera_to_initial_pose from open_drawer_RANSAC."
            )

        # Move the arm (this will set positions on the plant's context)
        move_wsg_camera_to_initial_pose(self.diagram, self.station, self.context)

        # Publish once so sensors (camera) generate outputs for the new pose
        self.diagram.ForcedPublish(self.context)

        # Read the camera point cloud from the diagram output port
        try:
            pc_port = self.diagram.GetOutputPort("camera0_point_cloud")
        except Exception as e:
            raise RuntimeError("Diagram has no output port 'camera0_point_cloud'.") from e

        pc = pc_port.Eval(self.context)
        self.camera_point_cloud = pc

        # Transition
        self.state = "open_drawer"
        return self.state

    def _enter_open_drawer(self) -> str:
        """Execute the drawer-opening sequence.
        
        Calls the open_drawer() function which handles all grasp detection,
        trajectory planning, controller setup, and simulation execution.
        The point cloud captured in preRANSAC is passed to open_drawer.
        """
        if self.camera_point_cloud is None:
            raise RuntimeError("Camera point cloud not available; run preRANSAC first.")

        # Plan trajectories using the perception result (no new diagram)
        traj_V_G, traj_wsg_command, switch_time, pull_params = plan_open_drawer(
            station=self.station, pc=self.camera_point_cloud, meshcat=self.meshcat,
            diagram_context=self.context
        )

        # Update the persistent TrajectorySource objects with the planned trajectories
        # (TrajectorySource provides UpdateTrajectory)
        try:
            self.V_G_source.UpdateTrajectory(traj_V_G)
        except Exception:
            # fallback: replace the source isn't possible in a built diagram; raise
            raise RuntimeError("Could not update V_G trajectory on persistent source")

        try:
            self.wsg_source.UpdateTrajectory(traj_wsg_command)
        except Exception:
            raise RuntimeError("Could not update WSG trajectory on persistent source")

        # Set the switch time on the VwgSwitcher instance
        # (the switcher stores it as a Python attribute)
        try:
            self.switcher._switch_time = switch_time
        except Exception:
            # best-effort; continue even if attribute assignment fails
            raise RuntimeWarning("Could not set switch_time on VwgSwitcher instance")

        # Initialize integrator to current IIWA positions
        plant = self.station_system.GetSubsystemByName("plant")
        plant_ctx = self.diagram.GetMutableSubsystemContext(plant, self.context)
        q0 = plant.GetPositions(plant_ctx, plant.GetModelInstanceByName("iiwa"))
        integ_ctx = self.integrator.GetMyContextFromRoot(self.context)
        self.integrator.set_integral_value(integ_ctx, q0)

        # Record current absolute time and advance the persistent simulator
        start_time = self.context.get_time()
        total_duration = traj_V_G.end_time() + 100.0
        target_time = start_time + total_duration
        print(f"[StateMachine.open_drawer] Advancing simulator from {start_time:.3f} to {target_time:.3f}")

        if self.meshcat:
            self.meshcat.StartRecording()
        self.simulator.AdvanceTo(target_time)
        if self.meshcat:
            self.meshcat.StopRecording()
            self.meshcat.PublishRecording()

        # Store home position (current initial IIWA config)
        # temp_context = self.station.CreateDefaultContext()
        # temp_plant_context = plant.GetMyContextFromRoot(temp_context)
        temp_plant_context = plant.GetMyContextFromRoot(self.context)
        iiwa_model = plant.GetModelInstanceByName("iiwa")
        self.home_q = plant.GetPositions(temp_plant_context, iiwa_model)

        self.state = "home"
        return self.state

    def _enter_home(self) -> str:
        """Return the arm to the initial home position using the persistent simulator.
        
        Updates the V_G trajectory source to a simple motion back to home position,
        then advances the persistent simulator.
        """
        if self.home_q is None:
            raise RuntimeError("Home position not captured; run open_drawer first.")
        
        plant = self.station.GetSubsystemByName("plant")
        # temp_context = self.station.CreateDefaultContext()
        # temp_plant_context = plant.GetMyContextFromRoot(temp_context)
        temp_plant_context = plant.GetMyContextFromRoot(self.context)
        
        # Current gripper pose from plant context
        wsg_body = plant.GetBodyByName("body")
        X_WG_current = plant.EvalBodyPoseInWorld(temp_plant_context, wsg_body)
        
        # Compute home gripper pose using forward kinematics
        plant.SetPositions(temp_plant_context, plant.GetModelInstanceByName("iiwa"), self.home_q)
        X_WG_home = plant.EvalBodyPoseInWorld(temp_plant_context, wsg_body)
        
        # Create trajectory from current to home (3 second duration)
        from pydrake.all import PiecewisePose
        sample_times = [0.0, 3.0]
        keyframes = [X_WG_current, X_WG_home]
        
        robot_position_trajectory = PiecewisePose.MakeLinear(sample_times, keyframes)
        traj_V_G = robot_position_trajectory.MakeDerivative()
        
        # Update the persistent V_G_source trajectory
        try:
            self.V_G_source.UpdateTrajectory(traj_V_G)
        except Exception as e:
            raise RuntimeError(f"Could not update V_G trajectory for home: {e}")
        
        # Update switcher to not switch (stay with trajectory throughout)
        # Home motion is just a simple trajectory, no compliant pull phase
        try:
            self.switcher._switch_time = traj_V_G.end_time() + 10.0  # Never switch
        except Exception:
            pass  # best-effort
        
        # Initialize integrator to current IIWA positions
        plant_ctx = self.diagram.GetMutableSubsystemContext(plant, self.context)
        q0 = plant.GetPositions(plant_ctx, plant.GetModelInstanceByName("iiwa"))
        integ_ctx = self.integrator.GetMyContextFromRoot(self.context)
        self.integrator.set_integral_value(integ_ctx, q0)
        
        # Advance simulator
        start_time = self.context.get_time()
        target_time = start_time + traj_V_G.end_time()
        print(f"[StateMachine.home] Returning to home position over {traj_V_G.end_time():.1f} seconds")
        
        if self.meshcat:
            self.meshcat.StartRecording()
        self.simulator.AdvanceTo(target_time)
        if self.meshcat:
            self.meshcat.StopRecording()
            self.meshcat.PublishRecording()
        
        print("[home] Arrived at home position")
        self.state = "done"
        return self.state

    def _create_new_builder_for_execution(self):
        """Helper to create a fresh DiagramBuilder for execution."""
        return DiagramBuilder()


def example_usage(meshcat=None):
    """Small helper showing how to use the state machine."""
    sm = StateMachine(meshcat=meshcat)
    print("initial state:", sm.state)
    sm.step()  # runs preRANSAC
    print("after preRANSAC state:", sm.state)
    sm.step()  # runs open_drawer
    print("after open_drawer state:", sm.state)
    sm.step()  # runs home
    print("after home state:", sm.state)
    # State machine is now in 'done' state
    return sm
