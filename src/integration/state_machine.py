from __future__ import annotations

from typing import Optional
import time

from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import Diagram, DiagramBuilder
from pydrake.all import PointCloud

from open_drawer import open_drawer, plan_open_drawer
import numpy as np
from pydrake.all import Integrator, Simulator, PiecewisePolynomial, TrajectorySource, ConstantVectorSource
from controller import PseudoInverseController, CompliantPullCommand, VwgSwitcher
from camera_system import CameraSystem
from perception import load_perception_assets, collect_multiview_data, cosine_average_task_clouds, bayesian_task_clouds
from grasp_planning import find_best_antipodal_grasp

device = "cpu"  # Change to "cpu" if no GPU is available.


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
        self.switcher = builder.AddSystem(VwgSwitcher(switch_time1=1.0, switch_time2=1000.0))
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


        self.assets = load_perception_assets(device=device)
        self.camera = CameraSystem(0, self.diagram, self.context)
        return

    def step(self) -> str:
        """Run the actions for the current state and (optionally)
        transition to the next state.
        preRANSAC -> open_drawer -> SAM

        Returns the name of the new state after stepping.
        """
        if self.state == "preRANSAC":
            return self._enter_preRANSAC()
        elif self.state == "open_drawer":
            return self._enter_open_drawer()
        elif self.state == "SAM":
            return self._enter_SAM()
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
        traj_V_G, traj_wsg_command, switch_time1, pull_params = plan_open_drawer(
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
            self.switcher._switch_time1 = switch_time1
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
        total_duration = traj_V_G.end_time() + 10.0
        target_time = start_time + total_duration
        print(f"[StateMachine.open_drawer] Advancing simulator from {start_time:.3f} to {target_time:.3f}")

        if self.meshcat:
            self.meshcat.StartRecording()
        self.simulator.AdvanceTo(target_time)
        # if self.meshcat:
        #     self.meshcat.StopRecording()
        #     self.meshcat.PublishRecording()

        self.state = "SAM"
        return self.state

    def _enter_SAM(self) -> str:
        """Move manipulator through q_checkpoints, capture point clouds at each pose.
        
        Similar to collect_multiview_data in perception.py, but uses the persistent
        simulator and directly accesses the plant context instead of SimulationState.
        """
        from pydrake.all import Concatenate, BaseField, Fields
        
        # Define checkpoint configurations for multiview capture
        q_checkpoints = np.array([
            # [0, 0, 0, 0, 0, 0, 0],  # default/current
            [-np.pi/4, np.pi/3, -np.pi/6, -np.pi/3, 0, np.pi/3, 0],         # checkpoint 1
            [-np.pi/3, np.pi/5, np.pi/6, -np.pi/4, 0, np.pi/4, np.pi/4],    # checkpoint 2
        ])
        
        # Configure VwgSwitcher for SAM mode: trajectory-only (no compliant pull)
        # Use absolute times: set switch_time2 to a very large value so we stay in trajectory mode
        current_time = self.context.get_time()
        self.switcher.set_switch_times(
            switch_time1= self.switcher._switch_time1,  # Don't switch at all during SAM
            switch_time2= current_time
        )
        # print(f"[_enter_SAM] Configured VwgSwitcher for trajectory-only (t={current_time:.1f} to t={current_time+sam_duration:.1f})")
        
        # Start MeshCat recording for SAM phase visualization
        # if self.meshcat:
        #     self.meshcat.StartRecording()
        #     print("[_enter_SAM] Started MeshCat recording")
        
        multiview_data = collect_multiview_data(
            integrator=self.integrator,
            diagram=self.diagram,
            context=self.context,
            simulator=self.simulator,
            camera=self.camera,
            q_checkpoints=q_checkpoints,
            assets=self.assets,
            dt=0.5,
            station_system=self.station_system,
            V_G_source=self.V_G_source,
            wsg_source=self.wsg_source,
        )
        
        # Stop recording and publish
        # if self.meshcat:
        #     self.meshcat.StopRecording()
        #     self.meshcat.PublishRecording()


        cosine_clusters, concat_pcd_mv, _ = cosine_average_task_clouds(
            multiview_data, self.camera, self.assets.tasks
        )
        bayes_clusters, _, probs = bayesian_task_clouds(
            multiview_data, self.camera, self.assets
        )

        grasp_pose = find_best_antipodal_grasp(bayes_clusters, concat_pcd_mv, self.meshcat)
        if grasp_pose is None:
            print("No grasp pose found. Check segmentation results and thresholds.")
            return

        print("Found feasible grasp pose:", grasp_pose)

        
        self.state = "done"
        return self.state

    def _create_new_builder_for_execution(self):
        """Helper to create a fresh DiagramBuilder for execution."""
        return DiagramBuilder()
