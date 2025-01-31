from .data_loader import DataLoader

class Buffer(object):
    # This class defines a buffer system that stores dataset information.
    # These information are needed to load data and apply grouping.

    def __init__(self):
        self.clear_buffer()
        return

    def clear_buffer(self):

        self.total_num_frames = 0
        self.has_video = True
        self.frame_width = 0
        self.frame_height = 0
        self.dataset = ""
        self.flag = 0  # subdataset id

        self.video_position_matrix = []
        self.video_velocity_matrix = []
        self.video_pedidx_matrix = []
        self.video_labels_matrix = []
        self.video_dynamics_matrix = []

        self.people_start_frame = []
        self.people_end_frame = []
        self.people_coords_complete = []
        self.people_velocity_complete = []

        self.H = []

        self.num_groups = 0

        self.if_processed_data = False
        self.if_processed_group = False        

        return

class Environment(object):
    # This class preloads the dataset into buffers
    # Then one is selected for each trial
    #
    # fps is the loaded fps, or 1 / dt

    def __init__(self, fps, dset_path, logger):
        self.env_dict = {}
        self.env = None
        self.fps = fps
        self.dset_path = dset_path
        self.logger = logger

        return
    
    def preload_data(self, envs):
        # envs is a list of tuples (env_name, env_flag)
        for env in envs:
            buffer = Buffer()
            data = DataLoader(env[0], env[1], self.fps, self.dset_path, self.logger)
            buffer = data.update_buffer(buffer)
            self.env_dict[env] = buffer
        return
    
    def select_env(self, env):
        # env is (env_name, env_flag)
        self.env = self.env_dict[env]
        return self.env
