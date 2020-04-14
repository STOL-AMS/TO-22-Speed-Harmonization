"""####################################################################################################################################################
##################################################### Traffic Monitor                             #####################################################
####################################################################################################################################################"""

import control_panel as cp
import vehicle_generator as vg
import numpy as np
import numba

#VEHICLE TRAJECTORY ATTRIBUTES
#   order of attributes as follows: [0: timestep, 1: vehicleid', 2: lane,
#   3: position, 4: speed, 5: gap, 6: delta_speed, 7: accelaration,
#   8: desired_speed, 9: lead_id, 10: lead_right_id, 11: lead_left_id,
#   12: pred_id, 13: pred_right_id, 14: pred_left_id, 15: lane_chg]
#   16: maximum_decelaration, 17: target_exit, 18: drv_bhvr, 19: pred_model_id
#   20: mon_timestep, 21: section, 22: vehicle_class     
    
#INITIAL TRAFFIC MONITORING ARRAY
#   order of attributes as follows:
#   0: mon_timestep, 1: section, 2: num_vehs 3: mean_speed, 4: density, 5: flow,
#   6: ssd , 7: mean_headway, 8: lane_changes, 9: acc_std, 10: ssd_slead,
#   11: mean_speed_slead, 12: mean_headway_slead, 13: congestion, 14: main_volume
#   15: ramp_volume, 16: SPDHRM, 17: LNDRP

@numba.jit(nopython=True)    
def init_tm_array():
    #   calculating number of sections
    num_section = int(cp.segment_length / cp.section_length)
    #   calculating number of monitoring timesteps
    mon_timestep_hz = cp.mon_timestep * 10
    num_mon_timestep = int(vg.sim_time_hz/mon_timestep_hz)  
    #   calculating number of rows for the array = (sections * mon_timesteps)
    tm_array_rows = num_section * num_mon_timestep  
    #   create array
    tm_array = np.zeros((tm_array_rows, 18), dtype = np.float32)
    #   enter main flow indicator
    tm_array[:,14] = cp.volume
    #   enter ramp flow indicator
    tm_array[:,15] = cp.ramp_volume_total
    #   enter SPDHRM indicator
    tm_array[:,16] = cp.SPDHRM_control
    #   enter lane drop indicator
    tm_array[:,17] = cp.ldrop_active
    
    k = 0
    for i in range(1, num_mon_timestep + 1):
        for j in range(1, num_section + 1):
            tm_array[k,0] = i
            tm_array[k,1] = j
            k += 1
    return tm_array

#   define numba function to create an array of unique vehicles
#   np.unique is not supported in numba
@numba.jit(nopython=True)
def numba_unique(veh_list):
    uniques = []
    #   if vehicle id is not found within list, add it
    for i in range(veh_list.shape[0]):
        found = 0
        for j in uniques:
            if(veh_list[i] == j):
                found = 1
        if(found == 0):
            uniques.append(veh_list[i])
    unique_array = np.array(uniques, dtype = np.float32)
    return unique_array

#   define function to calculate total distance moved by vehicles within section
@numba.jit(nopython=True)                
def edie_distance(in_veh_traj_section):
    unique_vehs = numba_unique(in_veh_traj_section[:,1])
    total_distance = 0
    if(len(unique_vehs > 0)):
        for j in unique_vehs:
            condition_unique = in_veh_traj_section[:,1] == j
            temp_dist = np.max(in_veh_traj_section[condition_unique,3])\
            - np.min(in_veh_traj_section[condition_unique,3])
            total_distance += temp_dist #meters
    return total_distance

#   define function to calculate total time spent by vehicles within section
@numba.jit(nopython=True)
def edie_time(in_veh_traj_section):
    unique_vehs = numba_unique(in_veh_traj_section[:,1])
    if(len(unique_vehs) > 0):
        total_time = len(in_veh_traj_section[:,1])
        total_time = total_time / 10 #convert timesteps to seconds
    return total_time

#VEHICLE TRAJECTORY ATTRIBUTES
#   order of attributes as follows: [0: timestep, 1: vehicleid', 2: lane,
#   3: position, 4: speed, 5: gap, 6: delta_speed, 7: accelaration,
#   8: desired_speed, 9: lead_id, 10: lead_right_id, 11: lead_left_id,
#   12: pred_id, 13: pred_right_id, 14: pred_left_id, 15: lane_chg]
#   16: maximum_decelaration, 17: current_gap, 18: current_delta_speed, 19: pred_model_id
#   20: mon_timestep, 21: section, 22: vehicle_class
      
#   define function to update traffic properties for connected vehicles   
@numba.jit(nopython=True)   
def update_tm_array_edie(in_tm_array, in_veh_traj, t):
    #   calculate current mon_timestep
    mon_timestep_to_update = t / (cp.mon_timestep * 10)
    #   finding the indeces to update
    index_start = np.where((in_tm_array[:,0] == mon_timestep_to_update) & \
                     (in_tm_array[:,1] == 1))
    index_start_value = int(index_start[0][0])
    num_section = int(cp.segment_length / cp.section_length)
    index_end_value = index_start_value + num_section
    #   loop over tm_array
    for i in range(index_start_value, index_end_value):
        section = in_tm_array[i,1]
        #   calculate properties for current section only, at this time step, and
        #   in main lanes, and for connected vehicles!
        condition_updt = (in_veh_traj[:,20] == mon_timestep_to_update) &\
        (in_veh_traj[:,21] == section) & (in_veh_traj[:,2] <= cp.num_lanes) &\
        ((in_veh_traj[:,22] == 1) | (in_veh_traj[:,22] == 2))
        #   to avoid error, calculate attributes if there are vehicles in those sections
        if(np.sum(in_veh_traj[condition_updt,21]) > 0):               
            #   calculate number of unique vehicles
            in_tm_array[i,2] = len(numba_unique(in_veh_traj[condition_updt,1]))
            #   calculate edie's mean speed in km/hr               
            in_tm_array[i,3] = edie_distance(in_veh_traj[condition_updt,])/ \
            edie_time(in_veh_traj[condition_updt,]) * (3600/1000)
            #   calculate density in veh/km
            in_tm_array[i,4] = edie_time(in_veh_traj[condition_updt,])/ \
            (cp.section_length * cp.mon_timestep) * 1000
            #   calculate flow
            in_tm_array[i,5] = in_tm_array[i,3] * in_tm_array[i,4]
            #   calculate ssd
            in_tm_array[i,6] = np.std(in_veh_traj[condition_updt,4])
            #   calculate mean_headway
            in_tm_array[i,7] = np.mean(in_veh_traj[condition_updt,5])
            #   calculate lane_changes
            in_tm_array[i,8] = np.sum(np.abs(in_veh_traj[condition_updt,15]))
            #   calculate acc std
            in_tm_array[i,9] = np.std(in_veh_traj[condition_updt,7])
    #   creating space lead variables
    for i in range(index_start_value, index_end_value):
        #   to avoid error at end of array for lead variables
        if(i < len(in_tm_array)):
            #   ssd_slead
            in_tm_array[i,10] = in_tm_array[i+1,6]
            #   mean_speed_slead
            in_tm_array[i,11] = in_tm_array[i+1,3]
            #   mean_headway_slead
            in_tm_array[i,12] = in_tm_array[i+1,7]
        #   setting lead variables at last section = same as current section
        if(in_tm_array[i,1] == num_section):
            in_tm_array[i,10] = in_tm_array[i,6]
            in_tm_array[i,11] = in_tm_array[i,3]
            in_tm_array[i,12] = in_tm_array[i,7]

#   define function to update traffic properties for ALL Vehicles  
@numba.jit(nopython=True)   
def update_tm_array_edie_all(in_tm_array, in_veh_traj, t):
    #   calculate current mon_timestep
    mon_timestep_to_update = t / (cp.mon_timestep * 10)
    #   finding the indeces to update
    index_start = np.where((in_tm_array[:,0] == mon_timestep_to_update) & \
                     (in_tm_array[:,1] == 1))
    index_start_value = int(index_start[0][0])
    num_section = int(cp.segment_length / cp.section_length)
    index_end_value = index_start_value + num_section
    #   loop over tm_array
    for i in range(index_start_value, index_end_value):
        section = in_tm_array[i,1]
        #   calculate properties for current section only, at this time step, and
        #   in main lanes, and for connected vehicles!
        condition_updt = (in_veh_traj[:,20] == mon_timestep_to_update) &\
        (in_veh_traj[:,21] == section) & (in_veh_traj[:,2] <= cp.num_lanes)
        #   to avoid error, calculate attributes if there are vehicles in those sections
        if(np.sum(in_veh_traj[condition_updt,21]) > 0):               
            #   calculate number of unique vehicles
            in_tm_array[i,2] = len(numba_unique(in_veh_traj[condition_updt,1]))
            #   calculate edie's mean speed in km/hr               
            in_tm_array[i,3] = edie_distance(in_veh_traj[condition_updt,])/ \
            edie_time(in_veh_traj[condition_updt,]) * (3600/1000)
            #   calculate density in veh/km
            in_tm_array[i,4] = edie_time(in_veh_traj[condition_updt,])/ \
            (cp.section_length * cp.mon_timestep) * 1000
            #   calculate flow
            in_tm_array[i,5] = in_tm_array[i,3] * in_tm_array[i,4]
            #   calculate ssd
            in_tm_array[i,6] = np.std(in_veh_traj[condition_updt,4])
            #   calculate mean_headway
            in_tm_array[i,7] = np.mean(in_veh_traj[condition_updt,5])
            #   calculate lane_changes
            in_tm_array[i,8] = np.sum(np.abs(in_veh_traj[condition_updt,15]))
            #   calculate acc std
            in_tm_array[i,9] = np.std(in_veh_traj[condition_updt,7])
    #   creating space lead variables
    for i in range(index_start_value, index_end_value):
        #   ssd_slead
        in_tm_array[i,10] = in_tm_array[i+1,6]
        #   mean_speed_slead
        in_tm_array[i,11] = in_tm_array[i+1,3]
        #   mean_headway_slead
        in_tm_array[i,12] = in_tm_array[i+1,7]
        #   setting lead variables at last section = same as current section
        if(in_tm_array[i,1] == num_section):
            in_tm_array[i,10] = in_tm_array[i,6]
            in_tm_array[i,11] = in_tm_array[i,3]
            in_tm_array[i,12] = in_tm_array[i,7]

#   define function to create travel time table (main lane vehicles only)
@numba.jit(nopython=True) 
def gen_travel_time(in_veh_trajs):
    #   filter vehicles that started and exited in main lanes
    cond = (in_veh_trajs[:,3] == 0) & (in_veh_trajs[:,2] <= cp.num_lanes) &\
    (in_veh_trajs[:,17] == 0)
    #   generate vector of vehicle ids
    vehs_main_lane = in_veh_trajs[cond,1]
    tt_vector = np.zeros(shape = (len(vehs_main_lane)), dtype = np.float32)
    veh_last_position = np.zeros(shape = (len(vehs_main_lane)), dtype = np.float32)
    #   loop over those ids and calculate travel time
    for i in range(len(vehs_main_lane)):
        veh_id = vehs_main_lane[i]
        cond_time_start = in_veh_trajs[:,1] == veh_id
        time_start = np.min(in_veh_trajs[cond_time_start,0])
        time_end = np.max(in_veh_trajs[cond_time_start,0])
        veh_last_position[i] = np.max(in_veh_trajs[cond_time_start,3])
        #   convert to seconds
        tt_vector[i] = (time_end - time_start) / 10
    #   filter vehicles that traversed whole segment
    cond_full_traverse = veh_last_position >= 5000
    vehs_main_lane = vehs_main_lane[cond_full_traverse]
    tt_vector = tt_vector[cond_full_traverse]
    tt_array = np.vstack((vehs_main_lane, tt_vector)).T
    return tt_array