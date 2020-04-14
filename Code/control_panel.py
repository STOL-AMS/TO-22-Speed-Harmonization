"""####################################################################################################################################################
##################################################### Control Panel                               #####################################################
####################################################################################################################################################"""

#Importing Python packages
import inspect
import os
import pandas as pd
import numpy as np

import i_o as io

#setting working directory
module_path = inspect.getfile(inspect.currentframe())
module_dir = os.path.realpath(os.path.dirname(module_path))

#   script version (mac = 1, windows = 0)
mac = 0

#   seed
seed = 10

#   simulation duration (min)
sim_time_min = 45

#   vehicle type market penetrations (SHOULD SUM TO 1)
mpr_RV = 0
mpr_CV = 1
mpr_AV = 0

#   load road geometry and flow file
if(mac == 0):   
    road_geometry = io.load_geometry('%s\\IO\\road_geometry.txt' %module_dir)
else:
    road_geometry = io.load_geometry('%s/IO/road_geometry.txt' %module_dir)

#   main lanes information
#   volume: veh per hour
#   segment_length: meter
volume, segment_length, num_lanes = io.main_lanes_info(road_geometry)

#   initial speed range (km/hr)
low_init_speed = 100
high_init_speed = 100
desired_speed = 100

#   on_ramp information
on_ramp_info = io.on_ramps_info(road_geometry)
ramp_volume_total = np.sum(on_ramp_info[:,4])
ramp_speed = 100
ramp_entry_time = 0 #min

#   off_ramp information
off_ramp_info = io.off_ramps_info(road_geometry)
distance_to_exit = 1000 #meters to start checking exit lanes
exit_active = 1

#   target_lane information
target_exit_list, target_exit_percentages = io.target_exit_info(road_geometry)

#   vehicle length (m)
vehicle_length = 5.001

#   minimum distance to enter simulation (m)
#min_gap = 2
min_gap = 2


#   Traffic Monitor settings
section_length = 200 #meters
mon_timestep = 10 #seconds

#   Predictive Speed Harmonization Settings

#   predictive model settings
SPDHRM_pred_horizon = 2
SPDHRM_state_identification = 'tti'

#   Centralized SPDHRM
#   SPDHRM_control_type: decision_tree or SSD (speed standard deviation)
#   1: decision_tree picks new speed based on current speed in section, empirical
#   2: SSD adjusts speeds of vehicles above/below 1 or 2 standard deviations of mean speed
#   SPDHRM_control_distance_type: fixed or variable
#   1: fixed broadcasting distance, 2: variable distance depending on vehicle speed, 3: fixed broadcasting point
SPDHRM_control = 0
SPDHRM_control_type = 1 # 1 tree, 2 ssd, 3 optimal_speed
SPDHRM_control_distance_type = 1 # 1: fixed-area, 2: variable, 3: fixed-point
SPDHRM_broadcasting_distance = 1000 # used for fixed distance only
SPDHRM_comp_error = 0   # 0.1 or 0.2
SPDHRM_ssd_threshold = 2 # used for SSD type only

#   Decentralized SPDHRM
dec_SPDHRM_control = 0
dec_SPDHRM_eval_distance = 5000 #m
dec_SPDHRM_control_distance = 500 #m
dec_SPDHRM_cong_check_type = 2 #'1: area or 2: section
dec_SPDHRM_CV_comm_range = 300 #m
dec_SPDHRM_multimodel = 3 #num of different prediction models
dec_SPDHRM_multimodel_id = [50] #[15, 30, 50]
dec_SPDHRM_multimodel_MPR = [1] #[0.3, 0.3, 0.4]

#   Optimization-based SPDHRM
opt_SPDHRM_control = 0
opt_SPDHRM_eval_duration = 30 #sec
opt_SPDHRM_speed_list = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55]
opt_SPDHRM_brdcst_dist_list = [500, 1000, 1500]
