
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

CARTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/linhai/app/isaac/isaac-sim-assets-1-4.1.0/Assets/Isaac/4.0/Isaac/Robots/Carter/carter_v1.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=2),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25), joint_pos={"left_wheel": 0.0, "right_wheel": 0.0,"rear_pivot": 0.0,"rear_axle": 0.0}
    ),
    actuators={
        "left_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel"],
            effort_limit=4000.0,
            velocity_limit=1000.0,
            stiffness=1.0,
            damping=10000.0,
        ),
        "right_wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel"],
            effort_limit=4000.0,
            velocity_limit=1000.0,
            stiffness=1.0,
            damping=10000.0
        ),
        "rear_pivot_actuator": ImplicitActuatorCfg(
            joint_names_expr=["rear_pivot"], effort_limit=4000.0, velocity_limit=1000.0, stiffness=1.0, damping=10000.0
        ),
        "rear_axle_actuator": ImplicitActuatorCfg(
            joint_names_expr=["rear_axle"], effort_limit=4000.0, velocity_limit=1000.0, stiffness=1.0, damping=10000.0
        ),
    },
)



