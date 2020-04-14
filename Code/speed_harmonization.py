"""###########################################################################
############## Predictive Speed Harmonization              ###################
############## control logic                               ###################
###########################################################################"""

import numpy as np
import control_panel as cp
import numba

###############################################################################                        
######################### CENTRALIZED Control Functions #######################
###############################################################################


#INITIAL TRAFFIC MONITORING ARRAY
#   order of attributes as follows:
#   0: mon_timestep, 1: section, 2: num_vehs 3: mean_speed, 4: density, 5: flow,
#   6: ssd , 7: mean_headway, 8: lane_changes, 9: acc_std, 10: ssd_slead,
#   11: mean_speed_slead, 12: mean_headway_slead, 13: congestion, 14: main_volume
#   15: ramp_volume, 16: SPDHRM, 17: LNDRP



#   initialize variables
congestion_location = 0
active_SPDHRM = 0
VSL_SPDHRM = cp.desired_speed
ssd_SPDHRM = 0

#   function to evaluate whether SPDHRM should be activated
def evaluate_SPDHRM(in_tm_array, in_veh_trajs_init, in_pred_model,
                    in_opt_spd_limit, t):
    #   STOP previous activation
    in_active_SPDHRM = 0
    in_VSL_SPDHRM = cp.desired_speed
    in_ssd_SPDHRM = 0
    #   First step: predict congestion location
    congestion_location = congestion_pred(in_tm_array, in_pred_model, t)[0]
    #   Second step: determine new desired speed
    if(congestion_location > 0):
        in_VSL_SPDHRM = det_new_speed(in_tm_array, congestion_location, in_opt_spd_limit, t)[0]
        in_ssd_SPDHRM = det_new_speed(in_tm_array, congestion_location, in_opt_spd_limit, t)[1]
        #   Third step: set SPDHRM activator to 1
        in_active_SPDHRM = 1
    return  [congestion_location, in_VSL_SPDHRM, in_active_SPDHRM, in_ssd_SPDHRM]

def congestion_pred(in_tm_array, in_pred_model, t):
    #   calculate monitoring timestep
    mon_timestep_to_update = t / (cp.mon_timestep * 10)
    #   finding the indeces to update
    index_start = np.where((in_tm_array[:,0] == mon_timestep_to_update) & \
                     (in_tm_array[:,1] == 1))
    index_start_value = int(index_start[0][0])
    index_end_value = index_start_value +\
    int(cp.segment_length / cp.section_length)

    #   creating array of explanatory variables
    #   train[['mean_speed_tlag10_slead1', 'ssd_tlag10_slead1',
    #          'mean_speed_tlag10', 'mean_headway_tlag10' ]]
    exp_variables = in_tm_array[index_start_value:index_end_value, [11,10,3,7]]
    congestion_pred_out = in_pred_model.predict(exp_variables)
    in_tm_array[index_start_value:index_end_value,13] = congestion_pred_out
    #   find section id of the start of congestion location
    cong_section_id = congestion_loc_fun(congestion_pred_out)
    #   calculate position of segment start (e.g. section 1 -> 0, section 2 _> 200)
    cong_x = cong_section_id * cp.section_length - cp.section_length
    return cong_section_id, cong_x

@numba.jit(nopython=True)                 
def congestion_loc_fun(in_congestion_pred):   
    found = 0
    cong_loc_tail = 0
    cong_size = 0
    cong_loc_head = 0
    #   loop over sections downstream to upstream (backward)
    #   congestion location is the the first predicted location downstream
    #   lowersing speed to solve first congestion will subsequently solve others downstream
    for i in range(len(in_congestion_pred)):
        if(found == 0):
            #   looking for the first congestion location
            if(in_congestion_pred[i] == 1):
                cong_loc_tail = i
                found = 1
        if(found == 1):
            if(in_congestion_pred[i] == 1):
                cong_size += 1
                cong_loc_head = cong_loc_tail + cong_size - 1
            else:
                #   -1 is because size includes the first section
                cong_loc_head = cong_loc_tail + cong_size - 1
                break
    return cong_loc_tail


#   define function to determine updated speed limit
#@numba.jit(nopython=True)
def det_new_speed(in_tm_array, in_cong_location, in_opt_spd_limit, t):
    mon_timestep_to_update = t / (cp.mon_timestep * 10)
    current_speed = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                (in_tm_array[:,1] == in_cong_location),3][0]
    current_ssd = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                (in_tm_array[:,1] == in_cong_location),6][0]
    speed_congestion = current_speed
    section_500 = max(1, in_cong_location - np.floor(500/cp.section_length))
    speed_500 = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                (in_tm_array[:,1] == section_500),3][0]
    section_1000 = max(1, in_cong_location - np.floor(1000/cp.section_length))
    speed_1000 = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                (in_tm_array[:,1] == section_1000),3][0] 
    section_1500 = max(1, in_cong_location - np.floor(1500/cp.section_length))
    speed_1500 = in_tm_array[(in_tm_array[:,0] == mon_timestep_to_update) &\
                                (in_tm_array[:,1] == section_1500),3][0] 
    
    if(cp.SPDHRM_control_type == 1):
        #   check speed limit tree
        if(current_speed < 55):
            new_speed = 55
        elif(current_speed < 70):
            new_speed = 75
        else: new_speed = 90        
    elif(cp.SPDHRM_control_type == 2):
        new_speed = current_speed
    elif(cp.SPDHRM_control_type == 3):
        input_data = np.array([speed_congestion, speed_500, speed_1000,
                               speed_1500]).reshape(1,-1)
        new_speed = in_opt_spd_limit.predict(input_data)[0]
    return [new_speed, current_ssd]

#VEHICLE TRAJECTORY ATTRIBUTES
#   order of attributes as follows: [0: timestep, 1: vehicleid', 2: lane,
#   3: position, 4: speed, 5: gap, 6: delta_speed, 7: accelaration,
#   8: desired_speed, 9: lead_id, 10: lead_right_id, 11: lead_left_id,
#   12: pred_id, 13: pred_right_id, 14: pred_left_id, 15: lane_chg]
#   16: maximum_decelaration, 17: target_exit, 18: drv_bhvr, 19: pred_model_id
#   20: mon_timestep, 21: section, 22: vehicle_class               

@numba.jit(nopython=True)
def update_speed(in_control_distance_type, in_congestion_location,
                 in_veh_trajs_init, in_veh_trajs, in_VSL_SPDHRM, in_ssd_SPDHRM,
                 in_veh_active_SPDHRM):
    #   update speed assuming FIXED control distance - area
    if(in_control_distance_type == 1):
        #   estimate location of speed control
		#	End of VSL at start of congestion section
        VSL_location_end = in_congestion_location * cp.section_length - cp.section_length
        VSL_location_start = max(VSL_location_end - cp.SPDHRM_broadcasting_distance,0)
        for i in range(len(in_veh_trajs_init)):
            #   set general desired speed (to undo previous SPDHRM after activation)
#            in_veh_trajs_init[i, 8] = cp.desired_speed
            #   update vehicles within broadcasting space
            if((in_veh_trajs_init[i,3] <= VSL_location_end) & \
               (in_veh_trajs_init[i,3] > VSL_location_start)):            
                #   control type decision_tree (1)
                if(cp.SPDHRM_control_type == 1):
                    #   update connected vehicles - with compliance error
                   if(in_veh_trajs_init[i,22] == 1):
                       in_veh_trajs_init[i, 8] = np.random.normal(in_VSL_SPDHRM,
                                       in_VSL_SPDHRM * cp.SPDHRM_comp_error)
                   #    update automated vehicles - no compliance error
                   elif(in_veh_trajs_init[i,22] == 2):
                       in_veh_trajs_init[i, 8] = in_VSL_SPDHRM
                       
                #   control type SSD (2)       
                elif(cp.SPDHRM_control_type == 2):
                    #   update connected vehicles only
                    if(in_veh_trajs_init[i,22] == 1):
                        #   update vehicles whose speed is within an SSD threshold
                        if((in_veh_trajs_init[i,4] <= in_VSL_SPDHRM - cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM) &\
                           (in_veh_trajs_init[i,4] >= in_VSL_SPDHRM + cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM)):
                            in_veh_trajs_init[i, 8] = np.random.normal(in_VSL_SPDHRM,
                                       in_VSL_SPDHRM * cp.SPDHRM_comp_error)
                    #   update connected vehicles only
                    if(in_veh_trajs_init[i,22] == 2):
                        #   update vehicles whose speed is within an SSD threshold
                        if((in_veh_trajs_init[i,4] <= in_VSL_SPDHRM - cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM) &\
                           (in_veh_trajs_init[i,4] >= in_VSL_SPDHRM + cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM)):
                            in_veh_trajs_init[i, 8] = in_VSL_SPDHRM
                            
    #   update speed assuming VARIABLE control distance                        
    elif(in_control_distance_type == 2):
        #   estimate location of speed control
        VSL_location_end = in_congestion_location * cp.section_length
        #   estimate x location of congestion start
        x_cong_start = in_congestion_location * cp.section_length - cp.section_length
        for i in range(len(in_veh_trajs_init)):
            #   update for positive vehicle ids (non stopped vehicles at ramps)
            if(in_veh_trajs_init[i, 1] > 0):
                #   set general desired speed (to undo previous SPDHRM after activation)
                in_veh_trajs_init[i, 8] = cp.desired_speed
                #    get current vehicle speed
                veh_speed = in_veh_trajs_init[i, 4]
                #    estimate distance traversed by vehicle (m)
                dist_trav = veh_speed * 1000/3600 * cp.mon_timestep  #convert to m/s
                #   get vehicle location
                veh_current_loc = in_veh_trajs_init[i, 3]
                #   estimate location after one full monitoring timestep
                veh_potential_loc = veh_current_loc + dist_trav
                #   estimate start location of 
                if((veh_potential_loc >  x_cong_start) &\
                   (veh_current_loc < x_cong_start)):
                    #   control type decision_tree (1)
                    if(cp.SPDHRM_control_type == 1):
                    #   update connected vehicles - with compliance error
                        if(in_veh_trajs_init[i,22] == 1):
                            in_veh_trajs_init[i, 8] = np.random.normal(in_VSL_SPDHRM,
                                             in_VSL_SPDHRM * cp.SPDHRM_comp_error)
                        #    update automated vehicles - no compliance error
                        elif(in_veh_trajs_init[i,22] == 2):
                            in_veh_trajs_init[i, 8] = in_VSL_SPDHRM
                            
                    #   control type SSD (2)
                    elif(cp.SPDHRM_control_type == 2):
                        #   update connected vehicles only
                        if(in_veh_trajs_init[i,22] == 1):
                            #   update vehicles whose speed is within an SSD threshold
                            if((in_veh_trajs_init[i,4] <= in_VSL_SPDHRM - cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM) &\
                               (in_veh_trajs_init[i,4] >= in_VSL_SPDHRM + cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM)):
                                in_veh_trajs_init[i, 8] = np.random.normal(in_VSL_SPDHRM,
                                           in_VSL_SPDHRM * cp.SPDHRM_comp_error)
                        #   update connected vehicles only
                        if(in_veh_trajs_init[i,22] == 2):
                            #   update vehicles whose speed is within an SSD threshold
                            if((in_veh_trajs_init[i,4] <= in_VSL_SPDHRM - cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM) &\
                               (in_veh_trajs_init[i,4] >= in_VSL_SPDHRM + cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM)):
                                in_veh_trajs_init[i, 8] = in_VSL_SPDHRM
                                
    #   update speed assuming FIXED control distance - point
    if(in_control_distance_type == 3):
        #   estimate location of speed control
        congestion_location_start = in_congestion_location * cp.section_length
        VSL_location_start = max(congestion_location_start - cp.SPDHRM_broadcasting_distance,0)
        for i in range(len(in_veh_trajs_init)):
            #   check only for non-stopping vehicles
            if((in_veh_trajs_init[i, 3] > 0) & (in_veh_trajs_init[i, 3] <= cp.segment_length) &\
               (in_veh_trajs_init[i, 2] <= cp.num_lanes)):
                #   set general desired speed (to undo previous SPDHRM after activation)
#                in_veh_trajs_init[i, 8] = cp.desired_speed
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
                    #   control type decision_tree (1)
                    if(cp.SPDHRM_control_type == 1):
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
#                        #   update connected vehicles - with compliance error
#                        if(in_veh_trajs_init[i,22] == 1):
#                           in_veh_trajs_init[i, 8] = np.random.normal(in_VSL_SPDHRM,
#                                           in_VSL_SPDHRM * cp.SPDHRM_comp_error)
#                        #    update automated vehicles - no compliance error
#                        elif(in_veh_trajs_init[i,22] == 2):
#                           in_veh_trajs_init[i, 8] = in_VSL_SPDHRM
                     
                    #   control type SSD (2)       
                    elif(cp.SPDHRM_control_type == 2):
                        #   update connected vehicles only
                        if(in_veh_trajs_init[i,22] == 1):
                            #   update vehicles whose speed is within an SSD threshold
                            if((in_veh_trajs_init[i,4] <= in_VSL_SPDHRM - cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM) &\
                               (in_veh_trajs_init[i,4] >= in_VSL_SPDHRM + cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM)):
                                in_veh_trajs_init[i, 8] = np.random.normal(in_VSL_SPDHRM,
                                           in_VSL_SPDHRM * cp.SPDHRM_comp_error)
                        #   update connected vehicles only
                        if(in_veh_trajs_init[i,22] == 2):
                            #   update vehicles whose speed is within an SSD threshold
                            if((in_veh_trajs_init[i,4] <= in_VSL_SPDHRM - cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM) &\
                               (in_veh_trajs_init[i,4] >= in_VSL_SPDHRM + cp.SPDHRM_ssd_threshold * in_ssd_SPDHRM)):
                                in_veh_trajs_init[i, 8] = in_VSL_SPDHRM
    return in_veh_trajs_init            
               
        

###############################################################################                        
####################### DECENTRALIZED Control Functions #######################
###############################################################################                
    
#INITIAL TRAFFIC MONITORING ARRAY
#   order of attributes as follows:
#   0: mon_timestep, 1: section, 2: num_vehs 3: mean_speed, 4: density, 5: flow,
#   6: ssd , 7: mean_headway, 8: lane_changes, 9: acc_std, 10: ssd_slead,
#   11: mean_speed_slead, 12: mean_headway_slead, 13: congestion, 14: main_volume
#   15: ramp_volume, 16: SPDHRM, 17: LNDRP
    
#INITIAL TRAFFIC MONITORING ARRAY MULTIMODEL
#   order of attributes as follows:
#   0: mon_timestep, 1: section, 2: pred_model_id, 3: num_vehs 4: mean_speed, 5: density, 6: flow,
#   7: ssd , 8: mean_headway, 9: lane_changes, 10: acc_std, 11: ssd_slead,
#   12: mean_speed_slead, 13: mean_headway_slead, 14: congestion, 15: main_volume
#   16: ramp_volume, 17: SPDHRM, 18: LNDRP

#VEHICLE TRAJECTORY ATTRIBUTES
#   order of attributes as follows: [0: timestep, 1: vehicleid', 2: lane,
#   3: position, 4: speed, 5: gap, 6: delta_speed, 7: accelaration,
#   8: desired_speed, 9: lead_id, 10: lead_right_id, 11: lead_left_id,
#   12: pred_id, 13: pred_right_id, 14: pred_left_id, 15: lane_chg]
#   16: maximum_decelaration, 17: target_exit, 18: drv_bhvr, 19: pred_model_id
#   20: mon_timestep, 21: section, 22: vehicle_class
    
def dec_eval_SPDHRM_v2(in_veh_trajs_init, in_veh_traj, in_tm_array,
                    in_pred_model, t):
    #   reset desired speed to deactivate previous SPDHRM
#    in_veh_trajs_init[:,8] = np.where((in_veh_trajs_init[:, 1] > 0), cp.desired_speed,
#                     in_veh_trajs_init[:,8])
#    in_veh_trajs_init[:,8] = dec_reset_speed(in_veh_trajs_init)
    #   estimate monitoring timestep to evaluate logic
    mon_timestep_to_update = t / (cp.mon_timestep * 10)
    #   predict congestion (vector 0/1 for all sections)
    congestion_pred, section_speeds = dec_pred_cong(mon_timestep_to_update,
                                    in_tm_array, in_pred_model)
    #   check congestion downstream for all vehicles
    cong_dwnstrm_loc_id, cong_dwnstrm_loc_speed =\
    dec_cong_info_downstream(in_veh_trajs_init, congestion_pred, section_speeds)
#    #   find mean speeds of all sections to evaluate new speed
#    cond_mon_timestep = in_tm_array[:,0] == mon_timestep_to_update
#    v_all_sections = in_tm_array[cond_mon_timestep, 3]
    #   update speed for vehicles with active speed harmonization
#    in_veh_trajs_init = dec_update_speed(in_veh_trajs_init, cong_downstream_check,
#                                         cong_downstream_loc_idx, v_all_sections)
    return cong_dwnstrm_loc_id, cong_dwnstrm_loc_speed


#   define function to predict congestion location in all sections with data
#   from connected vehicles
def dec_pred_cong(in_mon_timestep_to_update, in_tm_array, in_pred_model):
    #   finding the indeces to update
    index_start = np.where((in_tm_array[:,0] == in_mon_timestep_to_update) & \
                     (in_tm_array[:,1] == 1))
    index_start_value = int(index_start[0][0])
    index_end_value = index_start_value +\
    int(cp.segment_length / cp.section_length)    
    #   select variables to enter in prediction model at monitoring timestep
    exp_variables = in_tm_array[index_start_value:index_end_value, [11,10,3,7]]
    #   predict congestion for all sections    
    congestion_pred_out = in_pred_model.predict(exp_variables)  
    #   update congestion values
    in_tm_array[index_start_value:index_end_value,13] = congestion_pred_out
    #   extract section speeds to determine VSL later
    section_speeds = in_tm_array[index_start_value:index_end_value, 3]
    return congestion_pred_out, section_speeds


#   define function to determine the evaluation distance for each vehicle
#   evaluation distance is constrained by: 1) CV communication range, 2) CV MPR
@numba.jit(nopython=True) 
def dec_max_eval_distance(in_veh_position, in_veh_trajs_init):
    #   initial variable
    eval_dist_max = 0
    #   filter moving CAVs that are ahead of target vehicle in main lanes
    cond_filter = (in_veh_trajs_init[:,3] - in_veh_position >= 0) &\
    (in_veh_trajs_init[:, 3] <= cp.segment_length) &\
    (in_veh_trajs_init[:, 2] <= cp.num_lanes) &\
    ((in_veh_trajs_init[:,22] == 1) | (in_veh_trajs_init[:,22] == 2))
    CAVs_ahead = in_veh_trajs_init[cond_filter]
    #   calculate delta_x to see which vehicles are ahead of target vehicle
    delta_x = CAVs_ahead[:,3] - in_veh_position
    #   sort vehicles based on distance (returns indeces)
    sorted_idxs = np.argsort(delta_x)
    #   range is lenght -1 so that it doesn't raise an out of range error
    for i in range(len(sorted_idxs) - 1):
        #   find index of target vehicle to check
        idx_target = sorted_idxs[i]
        idx_lead = sorted_idxs[i+1]
        #   check vehicles ahead of target only
        #   find relative distance
        rel_distance = delta_x[idx_lead] - delta_x[idx_target]
        #   check if relative distance larger than communication range
        if(rel_distance <= cp.dec_SPDHRM_CV_comm_range):
            #   set maximum evaluation range as the position of the last
            #   connected vehicle within communication range
            eval_dist_max = CAVs_ahead[idx_lead,3] - in_veh_position
        else:
            break           
    return eval_dist_max
     

#   define function to check for each vehicle whether there is congestion
#   within control range (maximum of detection range and control range )
@numba.jit(nopython=True)   
def dec_cong_info_downstream(in_veh_trajs_init, in_congestion_pred, in_section_speeds):
    #   create a 0/1 vector to indicate a congestion within specified distance
    #   downstream - initial value is 0
    cong_dwnstrm_loc_id = np.zeros(shape = (len(in_veh_trajs_init)),
                                     dtype = np.float32)
    cong_dwnstrm_loc_speed = np.zeros(shape = (len(in_veh_trajs_init)),
                                     dtype = np.float32)
    for veh in range(len(in_veh_trajs_init)):
        #   apply logic to main lane (l < num_lanes) moving vehicles (0 < x < segment_length)
        #   to CAVs only
        if((in_veh_trajs_init[veh, 3] > 0) & (in_veh_trajs_init[veh, 3] <= cp.segment_length) &\
           (in_veh_trajs_init[veh, 2] <= cp.num_lanes) &\
           ((in_veh_trajs_init[veh, 22] == 1) | (in_veh_trajs_init[veh, 22] == 2))):
            
            veh_position = in_veh_trajs_init[veh, 3]
            max_eval_distance = dec_max_eval_distance(veh_position, in_veh_trajs_init)
            
            if(max_eval_distance > 0):
                num_sections_to_check = int(max_eval_distance/cp.section_length)            
                veh_section = int(in_veh_trajs_init[veh, 21])
                veh_section_idx = veh_section - 1
                start_idx = veh_section_idx + 1
                end_idx = min(start_idx + num_sections_to_check,
                              int(cp.segment_length / cp.section_length) - 1)
    
                #   find tail of congestion location downstream of vehicle
                for i in range(start_idx, end_idx + 1):
                    if(in_congestion_pred[i] == 1):
                        #   find section id
                        cong_dwnstrm_loc_id[veh] = i + 1
                        cong_dwnstrm_loc_speed[veh] = in_section_speeds[i]
                        break
                
#            #   veh_section index equals (section_id - 1)
#            if(cp.dec_SPDHRM_cong_check_type == 1):
#                check_congestion = np.sum(in_congestion_pred[start_idx:end_idx])
#            else:
#                if(in_congestion_pred[end_idx] == 1):
#                    check_congestion = 1
#            #   check if there is congestion in any section downstream
#            #   sum > 0
#            if (check_congestion > 0):
#                if(cp.dec_SPDHRM_cong_check_type == 1):
#                    cong_downstream_check[veh] = 1
#                    found = 0
#                    for i in range(start_idx, end_idx + 1):
#                        if(found == 0):
#                            if(in_congestion_pred[i] == 1):                
#                                cong_downstream_loc_idx[veh] = i
#                else:
#                    cong_downstream_loc_idx[veh] = end_idx
    return cong_dwnstrm_loc_id, cong_dwnstrm_loc_speed

#   define function to update speed for vehicles       
@numba.jit(nopython=True)     
def dec_update_speed(in_veh_trajs_init, in_cong_dwnstrm_loc_id, 
                     in_cong_dwnstrm_loc_speed, in_veh_active_SPDHRM):    
    for veh in range(len(in_veh_trajs_init)):
        #   STEP ONE: Determine VSL
        #   update moving vehicles in main lanes only if CONGESTION IS DETECTED
        cond_update = (in_veh_trajs_init[veh, 3] > 0) & (in_veh_trajs_init[veh, 3] <= cp.segment_length) &\
        (in_veh_trajs_init[veh, 2] <= cp.num_lanes) & (in_cong_dwnstrm_loc_id[veh] > 0)
        if(cond_update):
            #   reset speed to deactivate any previous VSL
#            in_veh_trajs_init[veh:8] = cp.desired_speed
            #   determine VSL
            if(in_cong_dwnstrm_loc_id[veh] > 0):
                section_speed = in_cong_dwnstrm_loc_speed[veh]
                if(section_speed < 55):
                    VSL = 55
                elif(section_speed < 70):
                    VSL = 75
                else: VSL = 90            
                
            #   STEP TWO: Determine VSL Start Point
            cong_loc_start = in_cong_dwnstrm_loc_id[veh] * cp.section_length - cp.section_length
            VSL_loc_start = max(cong_loc_start - cp.dec_SPDHRM_control_distance, 0)
            
            #   STEP THREE: Determine if Vehicle Crossed VSL Start Point
            #   get vehicle position in current timestep
            veh_position = in_veh_trajs_init[veh, 3]
            #   estimate distance traveled from previous timestep
            #   assumes same accelaration - faster than looking up value
            veh_acc = in_veh_trajs_init[veh, 7]
            veh_speed = in_veh_trajs_init[veh, 4] * 1000/3600
            distance_travelled = (veh_speed * 0.1) + (.5 * veh_acc * 0.1**2)
            veh_position_prev = veh_position - distance_travelled
            
            #   STEP FOUR: Update Vehicles that Crossed VSL Start            
            if((veh_position_prev < VSL_loc_start) & (veh_position >= VSL_loc_start)):
                #   set veh_active_SPDHRM_flag to 1 if vehicle crossed VSL Line
                in_veh_active_SPDHRM[veh] = 1
            #   update desired speed for flagged vehicles
            if(in_veh_active_SPDHRM[veh] == 1):
                #   update speed gradually over 200 m to avoid artificial slowdowns
                #   this is a car-following model issue not control related                        
                v_des = in_veh_trajs_init[veh, 8]
                delta_v_des_ratio = (v_des - VSL)/200
                if(veh_position < (VSL_loc_start + 200)):
                    VSL_update = v_des - delta_v_des_ratio * (veh_position - VSL_loc_start)
                else:
                    VSL_update = VSL
                #   update connected vehicles - with compliance error
                if(in_veh_trajs_init[veh,22] == 1):
                   in_veh_trajs_init[veh, 8] = np.random.normal(VSL_update,
                                   VSL_update * cp.SPDHRM_comp_error)
                #    update automated vehicles - no compliance error
                elif(in_veh_trajs_init[veh,22] == 2):
                   in_veh_trajs_init[veh, 8] = VSL_update            
    return in_veh_trajs_init