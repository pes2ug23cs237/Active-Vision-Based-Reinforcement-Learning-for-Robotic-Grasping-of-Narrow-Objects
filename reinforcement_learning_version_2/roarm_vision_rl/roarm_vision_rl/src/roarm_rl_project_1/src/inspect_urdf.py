"""
URDF Inspector - Debug robot structure
This script loads your robot and prints all joint/link information
"""
import pybullet as p
import pybullet_data
import os

def inspect_urdf():
    """Load URDF and print all joint/link information"""
    
    # Connect to PyBullet
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load robot
    robot_urdf_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'robot_files', 
        'roarm.urdf'
    )
    
    print("=" * 80)
    print(f"Loading URDF: {robot_urdf_path}")
    print("=" * 80)
    
    robot_id = p.loadURDF(robot_urdf_path, [0, 0, 0], useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    
    print(f"\nTotal joints found: {num_joints}")
    print("\n" + "=" * 80)
    print("JOINT/LINK INFORMATION:")
    print("=" * 80)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        
        joint_index = joint_info[0]
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        q_index = joint_info[3]
        u_index = joint_info[4]
        flags = joint_info[5]
        joint_damping = joint_info[6]
        joint_friction = joint_info[7]
        joint_lower_limit = joint_info[8]
        joint_upper_limit = joint_info[9]
        joint_max_force = joint_info[10]
        joint_max_velocity = joint_info[11]
        link_name = joint_info[12].decode('utf-8')
        joint_axis = joint_info[13]
        parent_frame_pos = joint_info[14]
        parent_frame_orn = joint_info[15]
        parent_index = joint_info[16]
        
        # Joint type names
        joint_type_names = {
            p.JOINT_REVOLUTE: "REVOLUTE",
            p.JOINT_PRISMATIC: "PRISMATIC",
            p.JOINT_SPHERICAL: "SPHERICAL",
            p.JOINT_PLANAR: "PLANAR",
            p.JOINT_FIXED: "FIXED"
        }
        type_name = joint_type_names.get(joint_type, f"UNKNOWN({joint_type})")
        
        print(f"\nJoint Index: {joint_index}")
        print(f"  Joint Name: {joint_name}")
        print(f"  Link Name: {link_name}")
        print(f"  Joint Type: {type_name}")
        print(f"  Parent Index: {parent_index}")
        
        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            print(f"  Limits: [{joint_lower_limit:.4f}, {joint_upper_limit:.4f}]")
            print(f"  Axis: {joint_axis}")
        
        if 'gripper' in link_name.lower() or 'tcp' in link_name.lower():
            print(f"  *** POTENTIAL END-EFFECTOR ***")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    # Find controllable joints
    controllable_joints = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        joint_name = joint_info[1].decode('utf-8')
        
        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            controllable_joints.append((i, joint_name))
    
    print(f"\nControllable Joints ({len(controllable_joints)}):")
    for idx, name in controllable_joints:
        print(f"  Index {idx}: {name}")
    
    # Find potential end-effector links
    print("\nPotential End-Effector Links:")
    found_tcp = False
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        link_name = joint_info[12].decode('utf-8')
        
        if 'tcp' in link_name.lower() or 'gripper' in link_name.lower() or 'hand' in link_name.lower():
            print(f"  Index {i}: {link_name}")
            found_tcp = True
    
    if not found_tcp:
        print(f"  No TCP/gripper link found. Using last link (index {num_joints-1})")
    
    print("\n" + "=" * 80)
    
    p.disconnect()


if __name__ == "__main__":
    inspect_urdf()