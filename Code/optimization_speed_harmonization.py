"""###########################################################################
############## Optimization-based Speed Harmonization      ###################
############## control logic                               ###################
###########################################################################"""

import numpy as np
import numba
import multiprocessing as mp
from multiprocessing import Process

import control_panel as cp
import speed_harmonization as sh
import driving_logic as dl

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

#   INITIALIZE VARIABLES
congestion_location = 0
brdcst_dist = 1000
VSL_SPDHRM = cp.desired_speed
active_SPDHRM = 0
speed_congestion = 0
speed_cong_slead = 0
ssd_cong_slead = 0
speed_500 = 0
speed_1000 = 0
speed_1500 = 0
flow_cong = 0
density_cong = 0


def opt_evaluate_SPDHRM(in_tm_array, in_pred_model,
                        in_timestep, in_eval_duration_sec,
                        in_veh_trajs_init,
                        in_cfm_TLPR, in_cfm_IDM, in_cfm_AV_AREM,
                        in_lcm_m, in_lcm_r):
    #   STOP previous activation
    in_active_SPDHRM = 0
    in_VSL_SPDHRM = cp.desired_speed
    in_brdcst_dist = 1000
    speed_congestion = 0
    speed_cong_slead = 0
    ssd_cong_slead = 0
    speed_500 = 0
    speed_1000 = 0
    speed_1500 = 0
    flow_cong = 0
    density_cong = 0
    
    #   First step: predict congestion location
    congestion_location = sh.congestion_pred(in_tm_array, in_pred_model, in_timestep)[0]
    #   Second step: determine new desired speed
    
    if(congestion_location > 0):
        in_brdcst_dist, in_VSL_SPDHRM =\
        opt_find_optimal_params(in_timestep, in_eval_duration_sec,
                               in_veh_trajs_init,
                               in_cfm_TLPR, in_cfm_IDM, in_cfm_AV_AREM,
                               in_lcm_m, in_lcm_r,
                               congestion_location)
        #   Third step: set SPDHRM activator to 1
        in_active_SPDHRM = 1
        
        #   find speed at 500, 1000, and 1500 meter before congestion location
        #   this is to approximate an updated speed decision-tree based on optimization results
        mon_timestep_to_update = in_timestep / (cp.mon_timestep * 10)
        speed_congestion = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                    (in_tm_array[:,1] == congestion_location),3][0]
        section_500 = max(1, congestion_location - np.floor(500/cp.section_length))
        speed_500 = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                    (in_tm_array[:,1] == section_500),3][0]
        section_1000 = max(1, congestion_location - np.floor(1000/cp.section_length))
        speed_1000 = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                    (in_tm_array[:,1] == section_1000),3][0] 
        section_1500 = max(1, congestion_location - np.floor(1500/cp.section_length))
        speed_1500 = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                    (in_tm_array[:,1] == section_1500),3][0]
        
        speed_cong_slead = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                    (in_tm_array[:,1] == congestion_location),11][0]
        ssd_cong_slead = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                    (in_tm_array[:,1] == congestion_location),10][0]
        
        flow_cong = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                    (in_tm_array[:,1] == congestion_location),5][0]
        density_cong = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                    (in_tm_array[:,1] == congestion_location),4][0]
    
    return  congestion_location, in_brdcst_dist, in_VSL_SPDHRM, in_active_SPDHRM, speed_congestion, speed_500, speed_1000, speed_1500, speed_cong_slead, ssd_cong_slead, flow_cong, density_cong




def opt_find_optimal_params(in_timestep, in_eval_duration_sec,
                       in_veh_trajs_init,
                       in_cfm_TLPR, in_cfm_IDM, in_cfm_AV_AREM,
                       in_lcm_m, in_lcm_r,
                       in_congestion_location):
    #   create array to store distance travelled values
    dt_array = np.zeros(shape = (len(cp.opt_SPDHRM_brdcst_dist_list) * len(cp.opt_SPDHRM_speed_list), 3),
                        dtype = np.dtype('f4'))
    #   initialize counter k to fill dt_array
    k = 0
    #   loop over combinations of speed and distance to determine optimal set
    for d in cp.opt_SPDHRM_brdcst_dist_list:
        for u in cp.opt_SPDHRM_speed_list:
            dist_traveled = opt_distance_traveled(in_timestep, in_eval_duration_sec,
                                                    in_veh_trajs_init,
                                                    in_cfm_TLPR, in_cfm_IDM, in_cfm_AV_AREM,
                                                    in_lcm_m, in_lcm_r,
                                                    d, u,
                                                    in_congestion_location)
            #   record solution set
            dt_array[k,0] = d
            dt_array[k,1] = u
            dt_array[k,2] = dist_traveled
            k += 1
    #   find optimal solution (maximum distance travelled)
    optimal_solution_index = np.argmax(dt_array[:,2])
    optimal_brdcst_dist = dt_array[optimal_solution_index, 0]
    optimal_speed = dt_array[optimal_solution_index, 1]
    return optimal_brdcst_dist, optimal_speed

def opt_distance_traveled(in_timestep, in_eval_duration_sec,
                       in_veh_trajs_init,
                       in_cfm_TLPR, in_cfm_IDM, in_cfm_AV_AREM,
                       in_lcm_m, in_lcm_r,
                       in_brdcst_dist, in_updt_speed,
                       in_congestion_location):    
    #   record vehicles current position
    veh_init_position = in_veh_trajs_init[:,3]
    #   create a copy of vehicle trajectories to be updated (simulated)
    update_veh_trajs_init = np.copy(in_veh_trajs_init)
    #   convert evaluation duration to hz (0.1 sec, +1 to include last value)
    eval_duration_hz = in_eval_duration_sec * 10
    #   initialize cent SPDHRM vectors
    veh_active_SPDHRM_temp = np.zeros(shape = (len(update_veh_trajs_init)),
                                 dtype = np.float32) #for fixed point cases
    for t in range(in_timestep, in_timestep + eval_duration_hz + 1):
        #   apply speed harmonization (in the first monitoring timestep only)
        if(t < in_timestep + cp.mon_timestep * 10):
            update_veh_trajs_init = opt_update_speed(update_veh_trajs_init, in_congestion_location,
                                                     in_brdcst_dist, in_updt_speed,
                                                     veh_active_SPDHRM_temp)
        #   reset speed
        elif(t == in_timestep + cp.mon_timestep * 10):
            update_veh_trajs_init[:,8] = sh.SPDHRM_reset_speed(update_veh_trajs_init)
        veh_to_update = dl.update_trajectories(update_veh_trajs_init,
                                               in_cfm_TLPR, in_cfm_IDM, in_cfm_AV_AREM,
                                               in_lcm_m, in_lcm_r, t)
        update_veh_trajs_init = dl.update_init_veh_trajs(update_veh_trajs_init,
                                                         veh_to_update)
    veh_final_position = update_veh_trajs_init[:,3]
    veh_distance = veh_final_position - veh_init_position
    total_veh_distance = np.sum(veh_distance)
    return total_veh_distance

@numba.jit(nopython=True)
def opt_update_speed(in_veh_trajs_init, in_congestion_location,
                 in_brdcst_dist, in_VSL_SPDHRM,
                 in_veh_active_SPDHRM):                                
    #   estimate location of speed control
    congestion_location_start = in_congestion_location * cp.section_length
    VSL_location_start = max(congestion_location_start - in_brdcst_dist, 0)
    for i in range(len(in_veh_trajs_init)):
        #   check only for non-stopping vehicles
        if((in_veh_trajs_init[i, 3] > 0) & (in_veh_trajs_init[i, 3] <= cp.segment_length) &\
           (in_veh_trajs_init[i, 2] <= cp.num_lanes)):
            #   get vehicle position in current timestep
            veh_position = in_veh_trajs_init[i, 3]
            #   estimate distance traveled from previous timestep
            #   assumes same accelaration - faster than looking up value
            veh_acc = in_veh_trajs_init[i, 7]
            veh_speed = in_veh_trajs_init[i, 4] * 1000/3600
            distance_travelled = (veh_speed * 0.1) + (.5 * veh_acc * 0.1**2)
            veh_position_prev = veh_position - distance_travelled
            #   update vehicle if it crossed broadcasting point
            if((veh_position_prev < VSL_location_start) & \
               (veh_position >= VSL_location_start)):
                in_veh_active_SPDHRM[i] = 1
            if(in_veh_active_SPDHRM[i] == 1):
                #   update speed gradually over 200 m to avoid artificial slowdowns
                #   this is a car-following model issue not control related                        
                v_des = in_veh_trajs_init[i, 8]
                delta_v_des_ratio = (v_des - in_VSL_SPDHRM)/200
                if(veh_position < (VSL_location_start + 200)):
                    VSL_update = v_des - delta_v_des_ratio * (veh_position - VSL_location_start)
                else:
                    VSL_update = in_VSL_SPDHRM
                #   update connected vehicles - with compliance error
                if(in_veh_trajs_init[i,22] == 1):
                   in_veh_trajs_init[i, 8] = np.random.normal(VSL_update,
                                   VSL_update * cp.SPDHRM_comp_error)
                #    update automated vehicles - no compliance error
                elif(in_veh_trajs_init[i,22] == 2):
                       in_veh_trajs_init[i, 8] = VSL_update
    return in_veh_trajs_init
