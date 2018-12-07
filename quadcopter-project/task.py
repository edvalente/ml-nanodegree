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
        # print(reward)
        actual_pos = self.sim.pose[:3]
        actual_x, actual_y, actual_z = self.sim.pose[:3]
        target_x, target_y, target_z = self.target_pos
        self.diff_x, self.diff_y, self.diff_z = abs(self.target_pos - self.sim.pose[:3])
        self.sqrt_z = np.sqrt(actual_z / target_z)
        self.pow_z = (actual_z / target_z) ** 2
        self.log_z = np.log(actual_z / target_z)
        self.sin_z = np.sin(actual_z / target_z * np.pi/2)
        self.euc_dist = np.sqrt(((self.sim.pose[:3] - self.target_pos)**2).sum())
        
        # Reward
        reward = self.pow_z
        reward = 150 - self.euc_dist
        if self.sim.v[2] > 0:
            reward += 25
        else:
            reward -= 25
            
        
            
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            self.previous_z = self.sim.pose[2]
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            self.stepped_z = self.sim.pose[2]
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state