import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 1.0 - 0.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # reward = np.tanh(reward) # normalize reward to [-1, 1]
        actual_x, actual_y, actual_z = self.sim.pose[:3]
        target_x, target_y, target_z = self.target_pos
        
        # Set euclidean distance
        diff_pose = (self.sim.pose[:3] - self.target_pos)**2
        self.euclidean_distance = np.sqrt(diff_pose.sum())
        
        # Starter reward for smaller euclidean distance
        reward = 1.5 * np.tanh(2 - self.euclidean_distance / 50)
        # reward = np.tanh(1.0 - 0.003*(abs(self.sim.pose[:3] - self.target_pos)).sum())
        
        # Reward for velocity_Z growing
        reward += 0.4 * np.tanh(self.sim.v[2] / 10)
        
        # Penalty if X is higher than absolute 1
        reward -= 0.5 * np.tanh(abs(actual_x/10))
        
        # Penalty if Y is higher than absolute 1
        reward -= 0.5 * np.tanh(abs(actual_y/10))
        
        # Reward if Z is higher than 0
        reward += 0.5 * np.tanh(actual_z/10)
        if actual_z < 5.0:
            reward -= 0.5
        else:
            reward += 0.5
            
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state