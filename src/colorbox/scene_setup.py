"""Utilities for constructing the Drake station used in the final project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pydrake.all import Context, Diagram, DiagramBuilder, Simulator, StartMeshcat
from pydrake.geometry import Box
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser as MBParser
from pydrake.multibody.plant import CoulombFriction
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.systems.primitives import ConstantVectorSource, Integrator, TrajectorySource

from manipulation.station import AddPointClouds, LoadScenario, MakeHardwareStation
import os

TABLE_SDF = """<?xml version=\"1.0\"?>
<sdf version=\"1.6\">
  <model name=\"table\">
    <link name=\"link\">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.0333</ixx>
          <iyy>0.0333</iyy>
          <izz>0.005</izz>
        </inertia>
      </inertial>
      <visual name=\"visual\">
        <geometry>
          <box>
            <size>2 2 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>0.7 0.7 0.7 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
        </material>
      </visual>
      <collision name=\"collision\">
        <geometry>
          <box>
            <size>2 2 0.1</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""


def _write_table_sdf(asset_dir: Path) -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = asset_dir / "table.sdf"
    path.write_text(TABLE_SDF)
    return path


def _write_bimanual_directives(asset_dir: Path, directives_dir: Path) -> Path:
    table_sdf = _write_table_sdf(asset_dir)
    directives_dir.mkdir(parents=True, exist_ok=True)
    directives = f"""directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
    X_PC:
        translation: [0, -0.5, 0]
        rotation: !Rpy {{ deg: [0, 0, 180] }}
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 90]}}
- add_model:
    name: table
    file: file://{table_sdf}
- add_weld:
    parent: world
    child: table::link
    X_PC:
        translation: [0.0, 0.0, -0.05]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
"""
    path = directives_dir / "bimanual_IIWA14_with_table_and_initials_and_assets.dmd.yaml"
    path.write_text(directives)
    return path


def _write_camera_directives(directives_dir: Path) -> Path:
    directives_dir.mkdir(parents=True, exist_ok=True)
    directives = """directives:
- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: wsg::body
        rotation: !Rpy { deg: [-90.0, 0.0, 0.0]}
        translation: [0.0, 0.0, 0.05]

- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: wsg::camera0_origin
    child: camera0::base
"""
    path = directives_dir / "camera_directives.dmd.yaml"
    path.write_text(directives)
    return path


def _write_cabinet_directives(directives_dir: Path, toolbox_path: Path) -> Path:
    directives_dir.mkdir(parents=True, exist_ok=True)
    directives = f"""directives:
- add_model:
    name: cabinet
    file: file:///{toolbox_path}
"""
    path = directives_dir / "cabinet_directives.dmd.yaml"
    path.write_text(directives)
    return path


def ensure_scenario_file(base_dir: Path | None = None) -> Path:
    """Generate all directive/scenario files and return the scenario path."""

    if base_dir is None:
        base_dir = Path.cwd()
    asset_dir = base_dir / "assets"
    directives_dir = base_dir / "directives"
    scenario_dir = base_dir / "scenarios"
    toolbox_urdf = base_dir / "toolbox" / "toolbox.urdf"

    bimanual = _write_bimanual_directives(asset_dir, directives_dir)
    camera = _write_camera_directives(directives_dir)
    cabinet = _write_cabinet_directives(directives_dir, toolbox_urdf)

    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario = scenario_dir / (
        "bimanual_IIWA14_with_table_and_initials_and_assets_and_cameras.scenario.yaml"
    )
    scenario.write_text(
        f"""directives:
    - add_directives:
        file: file://{bimanual}
    - add_directives:
        file: file://{camera}
    - add_directives:
        file: file://{cabinet}


cameras:
    camera0:
        name: camera0
        depth: True
        X_PB:
            base_frame: camera0::base

model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg
    wsg: !SchunkWsgDriver {{}}
"""
    )
    return scenario


def add_free_boxes_for_drawer(parser: MBParser, n_boxes: int = 3) -> None:
    """Add free-floating cubes near the toolbox before the plant finalizes."""

    plant = parser.plant()
    box_xyz = np.array([0.04, 0.04, 0.04])
    mass = 0.05

    unit_inertia = UnitInertia.SolidBox(*box_xyz)
    spatial_inertia = SpatialInertia(
        mass=mass,
        p_PScm_E=[0.0, 0.0, 0.0],
        G_SP_E=unit_inertia,
    )
    box_shape = Box(*box_xyz)
    friction = CoulombFriction(0.7, 0.5)

    colors = [
        np.array([1.0, 0.2, 0.2, 1.0]),
        np.array([0.2, 1.0, 0.2, 1.0]),
        np.array([0.2, 0.2, 1.0, 1.0]),
        np.array([1.0, 1.0, 0.2, 1.0]),
    ]

    z0 = 0.10
    x0 = 0.55
    y0 = 0.0
    offsets_xy = [(0.0, 0.0), (0.0, 0.06), (0.0, -0.06)]

    while len(offsets_xy) < n_boxes:
        offsets_xy.append(offsets_xy[len(offsets_xy) % 3])

    for i in range(n_boxes):
        model = plant.AddModelInstance(f"box_{i + 1}")
        body = plant.AddRigidBody("base", model, spatial_inertia)
        dx, dy = offsets_xy[i]
        p_WB = np.array([x0 + dx, y0 + dy, z0])
        plant.SetDefaultFreeBodyPose(body, RigidTransform(p_WB))

        plant.RegisterCollisionGeometry(
            body,
            RigidTransform(),
            box_shape,
            f"box_{i + 1}_collision",
            friction,
        )
        plant.RegisterVisualGeometry(
            body,
            RigidTransform(),
            box_shape,
            f"box_{i + 1}_visual",
            colors[i % len(colors)],
        )


@dataclass
class StationDiagram:
    """Container for the Drake diagram builder and helper systems."""

    meshcat: any
    builder: DiagramBuilder
    station_system: Diagram
    iiwa_source: ConstantVectorSource | None
    wsg_source: ConstantVectorSource | TrajectorySource
    velocity_source: ConstantVectorSource | TrajectorySource | None
    integrator: Integrator | None
    controller: any | None
    home_q: np.ndarray
    scenario_path: Path


def build_station_setup(
    meshcat=None,
    n_free_boxes: int = 3,
    use_velocity_control: bool = False,
    traj_V_G=None,
    traj_wsg_command=None,
) -> StationDiagram:
    """Create the DiagramBuilder that wraps the hardware station."""

    os.environ["MESHCAT_DEFAULT_PORT"] = "7000"

    scenario_path = ensure_scenario_file()
    if meshcat is None:
        meshcat = StartMeshcat()
    meshcat.Delete()

    scenario = LoadScenario(filename=str(scenario_path))
    builder = DiagramBuilder()

    def parser_cb(parser: MBParser):
        add_free_boxes_for_drawer(parser, n_boxes=n_free_boxes)

    station = MakeHardwareStation(
        scenario,
        meshcat=meshcat,
        parser_prefinalize_callback=parser_cb,
    )
    station_system = builder.AddSystem(station)

    iiwa_port = station_system.GetInputPort("iiwa.position")
    q_const = np.zeros(iiwa_port.size())
    q_const[0] = -3 * np.pi / 8.0
    q_const[1] = np.pi / 4.0
    q_const[3] = -2 * np.pi / 6.0
    q_const[5] = 3 * np.pi / 6.0
    q_const[6] = np.pi / 2.0

    iiwa_source = None
    velocity_source = None
    integrator = None
    controller = None
    if use_velocity_control:
        from .controllers import PseudoInverseController

        velocity_source = (
            builder.AddSystem(TrajectorySource(traj_V_G))
            if traj_V_G is not None
            else builder.AddSystem(ConstantVectorSource(np.zeros(6)))
        )
        controller = builder.AddSystem(
            PseudoInverseController(station_system.GetSubsystemByName("plant"))
        )
        integrator = builder.AddSystem(Integrator(7))

        builder.Connect(velocity_source.get_output_port(), controller.GetInputPort("V_WG"))
        builder.Connect(
            station_system.GetOutputPort("iiwa.position_measured"),
            controller.GetInputPort("iiwa.position"),
        )
        builder.Connect(controller.get_output_port(), integrator.get_input_port())
        builder.Connect(integrator.get_output_port(), iiwa_port)
    else:
        iiwa_source = builder.AddSystem(ConstantVectorSource(q_const))
        builder.Connect(iiwa_source.get_output_port(), iiwa_port)

    wsg_port = station_system.GetInputPort("wsg.position")
    if traj_wsg_command is not None and use_velocity_control:
        wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))
    else:
        wsg_opening = np.full(wsg_port.size(), 0.05)
        wsg_source = builder.AddSystem(ConstantVectorSource(wsg_opening))
    builder.Connect(wsg_source.get_output_port(), wsg_port)

    pc_ports = AddPointClouds(
        scenario=scenario,
        builder=builder,
        station=station_system,
        meshcat=meshcat,
    )
    for name, sys in pc_ports.items():
        builder.ExportOutput(sys.point_cloud_output_port(), f"{name}_point_cloud")

    return StationDiagram(
        meshcat=meshcat,
        builder=builder,
        station_system=station_system,
        iiwa_source=iiwa_source,
        wsg_source=wsg_source,
        velocity_source=velocity_source,
        integrator=integrator,
        controller=controller,
        home_q=q_const,
        scenario_path=scenario_path,
    )


@dataclass
class SimulationState:
    """Holds the built diagram and helper accessors for simulation."""

    meshcat: any
    station: Diagram
    diagram: Diagram
    simulator: Simulator
    diagram_context: Context
    iiwa_source: ConstantVectorSource | None
    wsg_source: ConstantVectorSource | TrajectorySource
    velocity_source: ConstantVectorSource | TrajectorySource | None = None
    integrator: Integrator | None = None
    controller: any | None = None
    home_q: np.ndarray | None = None
    time: float = 0.0

    @property
    def station_context(self) -> Context:
        return self.diagram.GetMutableSubsystemContext(self.station, self.diagram_context)

    @property
    def plant(self):
        return self.station.GetSubsystemByName("plant")

    @property
    def plant_context(self) -> Context:
        return self.diagram.GetMutableSubsystemContext(self.plant, self.diagram_context)

    def advance(self, duration: float) -> None:
        target = self.time + duration
        self.simulator.AdvanceTo(target)
        self.time = target

    def set_arm_configuration(self, q: Sequence[float]) -> None:
        """Set IIWA joint positions; works for position or velocity-control setups."""
        q_arr = np.asarray(q)
        if q_arr.shape[0] != 7:
            # Extract the iiwa portion if a full generalized-position vector is passed.
            q_arr = self.plant.GetPositionsFromArray(self.plant.GetModelInstanceByName("iiwa"), q_arr)

        if self.iiwa_source is not None:
            src_context = self.iiwa_source.GetMyContextFromRoot(self.diagram_context)
            self.iiwa_source.get_mutable_source_value(src_context).set_value(q_arr)
        elif self.integrator is not None:
            integ_ctx = self.integrator.GetMyContextFromRoot(self.diagram_context)
            self.integrator.set_integral_value(integ_ctx, q_arr)
            # Keep plant context in sync for immediate queries (camera, perception).
            plant_ctx = self.plant_context
            iiwa_model = self.plant.GetModelInstanceByName("iiwa")
            self.plant.SetPositions(plant_ctx, iiwa_model, q_arr)
            self.diagram.ForcedPublish(self.diagram_context)
        else:
            raise RuntimeError("No mechanism available to set arm configuration.")

    def set_gripper_opening(self, opening: float) -> None:
        """Set the target opening of the Schunk WSG gripper fingers (meters)."""
        src_context = self.wsg_source.GetMyContextFromRoot(self.diagram_context)
        value = self.wsg_source.get_mutable_source_value(src_context).get_mutable_value()
        value[:] = opening

    def set_gripper_twist(self, V_WG: Sequence[float]) -> None:
        """Command spatial velocity of the gripper frame when velocity control is enabled."""
        if self.velocity_source is None:
            raise RuntimeError("Velocity control not enabled for this SimulationState.")
        src_context = self.velocity_source.GetMyContextFromRoot(self.diagram_context)
        self.velocity_source.get_mutable_source_value(src_context).set_value(V_WG)


def build_simulation(setup: StationDiagram) -> SimulationState:
    """Finalize the builder into a diagram and simulator."""

    diagram = setup.builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, diagram_context)
    simulator.Initialize()
    diagram.ForcedPublish(diagram_context)
    if setup.integrator is not None and setup.home_q is not None:
        integ_context = setup.integrator.GetMyContextFromRoot(diagram_context)
        setup.integrator.set_integral_value(integ_context, setup.home_q)
    return SimulationState(
        meshcat=setup.meshcat,
        station=setup.station_system,
        diagram=diagram,
        simulator=simulator,
        diagram_context=diagram_context,
        iiwa_source=setup.iiwa_source,
        wsg_source=setup.wsg_source,
        velocity_source=setup.velocity_source,
        integrator=setup.integrator,
        controller=setup.controller,
        home_q=setup.home_q,
    )


def initialize_drawer_boxes(
    sim_state: SimulationState, offsets: Sequence[np.ndarray] | None = None
) -> None:
    """Place the floating cubes inside the third drawer of the toolbox."""

    if offsets is None:
        offsets = [
            np.array([0.00, -0.10, -0.15]),
            np.array([0.10, -0.10, -0.15]),
            np.array([-0.10, -0.10, -0.15]),
        ]

    plant = sim_state.plant
    plant_context = sim_state.plant_context
    cabinet = plant.GetModelInstanceByName("cabinet")
    drawer_frame = plant.GetFrameByName("large_drawer_3", cabinet)
    X_WD = plant.EvalBodyPoseInWorld(plant_context, drawer_frame.body())

    for i, p_DB in enumerate(offsets, start=1):
        box_model = plant.GetModelInstanceByName(f"box_{i}")
        box_body = plant.GetBodyByName("base", box_model)
        plant.SetFreeBodyPose(plant_context, box_body, X_WD @ RigidTransform(p_DB))
