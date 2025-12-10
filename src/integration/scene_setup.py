from pathlib import Path
from pydrake.all import Context, Diagram, DiagramBuilder, Simulator, StartMeshcat
from manipulation.station import AddPointClouds, LoadScenario, MakeHardwareStation
import numpy as np
from pydrake.systems.primitives import ConstantVectorSource


# Table SDF string
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

# Writes a directives file for a bimanual IIWA14 setup with a table.
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

# Writes a directives file to mount a camera on the gripper.
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

# Writes a directives file for adding a cabinet with a drawer.
def _write_cabinet_directives(directives_dir: Path, toolbox_path: Path) -> Path:
    directives_dir.mkdir(parents=True, exist_ok=True)
    directives = f"""directives:
- add_model:
    name: drawer
    file: file:///{toolbox_path}
    default_joint_positions:
        base_drawer_joint: [0.0]
- add_weld:
    parent: world
    child: drawer::base_link
    X_PC:
        translation: [0.0, 1.05, 0.2]
        rotation: !Rpy {{ deg: [0, 0, -90] }}
"""
    path = directives_dir / "cabinet_directives.dmd.yaml"
    path.write_text(directives)
    return path

# Creates a scenario file that includes the bimanual IIWA14 setup, camera, and cabinet.
def scenario_file(base_dir: Path | None = None) -> Path:
    if base_dir is None:
        base_dir = Path.cwd()
    asset_dir = base_dir / "assets"
    directives_dir = base_dir / "directives"
    scenario_dir = base_dir / "scenarios"
    cabinet_urdf = base_dir / "drawer.urdf"

    bimanual = _write_bimanual_directives(asset_dir, directives_dir)
    camera = _write_camera_directives(directives_dir)
    cabinet = _write_cabinet_directives(directives_dir, cabinet_urdf)

    scenario_dir.mkdir(parents=True, exist_ok=True)

    scenario = scenario_dir / (
        "everything.scenario.yaml"
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


def build_station_setup(
        meshcat=None,
        connect_default_iiwa: bool = True,
        connect_default_wsg: bool = True,
):

    scenario_path = scenario_file()
    if meshcat is None:
        meshcat = StartMeshcat()
        print()
    meshcat.Delete()

    scenario = LoadScenario(filename=str(scenario_path))
    builder = DiagramBuilder()
    station = MakeHardwareStation(
        scenario=scenario,
        meshcat=meshcat,
    )

    station_system = builder.AddSystem(station)

    # When everything is static
    iiwa_port = station_system.GetInputPort("iiwa.position")
    q_const = np.zeros(iiwa_port.size())
    q_const[0] = -3 * np.pi / 8.0
    q_const[1] = np.pi / 4.0
    q_const[3] = -2 * np.pi / 6.0
    q_const[5] = 3 * np.pi / 6.0
    q_const[6] = np.pi / 2.0

    # Only add default IIWA position source if requested. Callers that
    # want to provide a persistent controller should set
    # connect_default_iiwa=False and add their own sources/controllers.
    if connect_default_iiwa:
        iiwa_source = builder.AddSystem(ConstantVectorSource(q_const))
        builder.Connect(iiwa_source.get_output_port(), iiwa_port)

    wsg_port = station_system.GetInputPort("wsg.position")
    # Default WSG opening; optional to skip so callers may add their own
    # gripper command source.
    if connect_default_wsg:
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

    return builder, station, meshcat