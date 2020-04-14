"""#####################################################################################################################################################
##################################################### Northwestern University                     ######################################################
##################################################### CAV ML Microsimulator                       ######################################################
##################################################### Amr Elfar & Alireza Talebpour               ######################################################
##################################################### Start Date: Feb 23, 2018                    ######################################################
##################################################### Update: 11/08/2018                          ######################################################
#####################################################################################################################################################"""

#Importing Python packages
import pandas as pd
import numpy as np
#from tqdm import tqdm #progress bar package
import inspect
import os

import vehicle_generator as vg
import car_following_models as cf
import driving_logic as dl
import lane_changing_models as lc
import control_panel as cp
import traffic_monitor as tm
import visualization as vs
import traffic_prediction as tp
import speed_harmonization as sh
import optimization_speed_harmonization as osh

#   this line is added to protect against multiprocessing error in windows
if __name__ == '__main__':

    #setting working directory
    module_path = inspect.getfile(inspect.currentframe())
    module_dir = os.path.realpath(os.path.dirname(module_path))
    os.chdir(module_dir)
    
    print(module_dir)
    
    #Initialization
    #   load generated vehicles
    veh_trajs_init = vg.all_trajs
    #   Load master trajectory array
    veh_trajs = vg.veh_trajs
    #   load TALEBPOUR model
    cfm_TLPR = cf.TALEBPOUR(cp.tlpr_acc_max, cp.tlpr_dec_max, cp.tlpr_crash_duration)
    #   load IDM car following model
    cfm_IDM = cf.IDM(cp.idm_a, cp.idm_b, cp.idm_T, cp.idm_so, cp.idm_delta)
    #   load AV_AREM model
    cfm_AV_AREM = cf.AV_AREM(cp.av_acc_max, cp.av_dec_max, cp.av_isolated_sensor_range,
                             cp.av_k, cp.av_ka, cp.av_kv, cp.av_kd, cp.av_react)
    
    #   load MOBIL lane changing model for main lanes
    lcm_m = lc.MOBIL(cp.mobil_m_b_safe, cp.mobil_m_p, cp.mobil_m_a_thresh)
    #   load MOBIL lane changing model for ramps
    lcm_r = lc.MOBIL(cp.mobil_r_b_safe, cp.mobil_r_p, cp.mobil_r_a_thresh)
    #   load initial traffic monitoring matrix for connected vehicles
    tm_array = tm.init_tm_array()
    #   load initial traffic monitoring matrix for MULTIMODE case
    if((cp.dec_SPDHRM_control == 1) & (cp.dec_SPDHRM_multimodel > 0)):
        tm_array_multimodel = tm.init_tm_array_multimodel()
    #   load independent traffic monitoring for all vehicles
    tm_array_all = tm.init_tm_array()
    #   Load base prediction model
    off_model = tp.offline_model()
    
    #   INITIALIZE SPDHRM
    if(cp.SPDHRM_control == 1):
        #   initialize cent SPDHRM vectors
        veh_active_SPDHRM = np.zeros(shape = (len(veh_trajs_init)),
                                         dtype = np.float32) #for fixed point cases
        #   Train base model using historical (base) data
        pred_model = off_model.base_model_simple(cp.SPDHRM_state_identification,
                                                 cp.SPDHRM_pred_horizon)
        #   Train model to select optimal speed limit based on optimization simulations
        opt_spd_limit = tp.optimal_spd_pred()
        
    elif(cp.dec_SPDHRM_control == 1):
        #   initialize dec SPDHRM vectors
        cong_dwnstrm_loc_id = np.zeros(shape = (len(veh_trajs_init)),
                                         dtype = np.float32)
        cong_dwnstrm_loc_speed = np.zeros(shape = (len(veh_trajs_init)),
                                         dtype = np.float32)
        veh_active_SPDHRM = np.zeros(shape = (len(veh_trajs_init)),
                                         dtype = np.float32)
        if(cp.dec_SPDHRM_multimodel > 0):
            pred_model = off_model.base_model_simple_multimodel(cp.SPDHRM_state_identification,
                                          cp.SPDHRM_pred_horizon)
        else:
            pred_model = off_model.base_model_simple(cp.SPDHRM_state_identification,
                                              cp.SPDHRM_pred_horizon)
    elif(cp.opt_SPDHRM_control == 1):
        #   initialize opt SPDHRM vectors
        veh_active_SPDHRM = np.zeros(shape = (len(veh_trajs_init)),
                                         dtype = np.float32) #for fixed point cases
        #   Train base model using historical (base) data
        pred_model = off_model.base_model_simple(cp.SPDHRM_state_identification,
                                                 cp.SPDHRM_pred_horizon)
    
    #initialize track vectors for diagnostic purposes       
    track_activeSPDHRM = []
    track_timestep = []
    track_VSL = []
    track_brdcst_dist = []
    track_spd_congestion = []
    track_spd_500 = []
    track_spd_1000 = []
    track_spd_1500 = []
    track_speed_cong_slead = []
    track_ssd_cong_slead = []
    track_flow_cong = []
    track_density_cong = []
    
    
    #   start a run
    #   main time loop
#    for timesteps in tqdm(range(0, vg.sim_time_hz + 1)):
    for timesteps in range(0, vg.sim_time_hz + 1):
        #   update vehicle trajectories
        #   perform car following and lane changing
        veh_to_update = dl.update_trajectories(veh_trajs_init, cfm_TLPR, cfm_IDM, cfm_AV_AREM,
                                               lcm_m, lcm_r, timesteps)
        #   assign updated trajectories to master list
        veh_trajs = dl.assign_trajs(veh_trajs, veh_to_update)
        #   update init trajectory list
        veh_trajs_init = dl.update_init_veh_trajs(veh_trajs_init,veh_to_update)
        #   update the following at each monitoring timestep 1 second is 10 timesteps
        if((timesteps % (cp.mon_timestep * 10) == 0) & (timesteps > 0)):
            #   update traffic monitor
            tm.update_tm_array_edie(tm_array, veh_trajs, timesteps)
            tm.update_tm_array_edie_all(tm_array_all, veh_trajs, timesteps)
            if((cp.dec_SPDHRM_control == 1) & (cp.dec_SPDHRM_multimodel > 0)):
                tm.update_tm_array_edie_multimodel(tm_array_multimodel, veh_trajs,
                                                   cp.dec_SPDHRM_multimodel_id, timesteps)
            #   evaluate DECENTRALIZED speed harmonization
            if(cp.dec_SPDHRM_control == 1):
                #   reset desired speed to deactivate previous SPDHRM
                veh_trajs_init[:,8] = sh.SPDHRM_reset_speed(veh_trajs_init)
                #   reset vehicle SPDHRM active flag
                veh_active_SPDHRM[:] = 0.0
                if(cp.dec_SPDHRM_multimodel > 0):
                    cong_dwnstrm_loc_id, cong_dwnstrm_loc_speed = sh.dec_eval_SPDHRM_multimodel(veh_trajs_init, veh_trajs,
                                                   tm_array_multimodel, pred_model, timesteps)
                else:
                    cong_dwnstrm_loc_id, cong_dwnstrm_loc_speed = sh.dec_eval_SPDHRM_v2(veh_trajs_init, veh_trajs,
                                                       tm_array, pred_model, timesteps)
            #   evaluate CENTRALIZED speed harmonization (if activated in control panel)
            if(cp.SPDHRM_control == 1):
                #   reset desired speed to deactivate previous SPDHRM
                veh_trajs_init[:,8] = sh.SPDHRM_reset_speed(veh_trajs_init)
                #   reset vehicle SPDHRM active flag
                veh_active_SPDHRM[:] = 0.0
                sh.congestion_location, sh.VSL_SPDHRM, sh.active_SPDHRM, sh.ssd_SPDHRM =\
                sh.evaluate_SPDHRM(tm_array, veh_trajs_init, pred_model, opt_spd_limit, timesteps)
                
                track_timestep.append(timesteps)
                track_activeSPDHRM.append(sh.active_SPDHRM)
                track_VSL.append(sh.VSL_SPDHRM)
                
            #   evaluate OPTIMIZATION-based speed harmonization (if activated in control panel)
            if(cp.opt_SPDHRM_control == 1):
                #   reset desired speed to deactivate previous SPDHRM
                veh_trajs_init[:,8] = sh.SPDHRM_reset_speed(veh_trajs_init)
                #   reset vehicle SPDHRM active flag
                veh_active_SPDHRM[:] = 0.0
                osh.congestion_location, osh.brdcst_dist, osh.VSL_SPDHRM, osh.active_SPDHRM, \
                osh.speed_congestion, osh.speed_500, osh.speed_1000, osh.speed_1500, \
                osh.speed_cong_slead, osh.ssd_cong_slead, osh.flow_cong, osh.density_cong =\
                osh.opt_evaluate_SPDHRM(tm_array, pred_model,
                                        timesteps, cp.opt_SPDHRM_eval_duration,
                                        veh_trajs_init,
                                        cfm_TLPR, cfm_IDM, cfm_AV_AREM,
                                        lcm_m, lcm_r)
                
                track_timestep.append(timesteps)
                track_activeSPDHRM.append(osh.active_SPDHRM)
                track_VSL.append(osh.VSL_SPDHRM)
                track_brdcst_dist.append(osh.brdcst_dist)
                track_spd_congestion.append(osh.speed_congestion)
                track_spd_500.append(osh.speed_500)
                track_spd_1000.append(osh.speed_1000)
                track_spd_1500.append(osh.speed_1500)
                track_speed_cong_slead.append(osh.speed_cong_slead)
                track_ssd_cong_slead.append(osh.ssd_cong_slead)
                track_flow_cong.append(osh.flow_cong)
                track_density_cong.append(osh.density_cong)
                
        #   update speed if speed harmonization is active
        if(sh.active_SPDHRM == 1):
            veh_trajs_init = sh.update_speed(cp.SPDHRM_control_distance_type, sh.congestion_location,
                            veh_trajs_init, veh_trajs, sh.VSL_SPDHRM, sh.ssd_SPDHRM, veh_active_SPDHRM)
        if(cp.dec_SPDHRM_control == 1):
            veh_trajs_init = sh.dec_update_speed(veh_trajs_init, cong_dwnstrm_loc_id,
                                                 cong_dwnstrm_loc_speed, veh_active_SPDHRM)
        if(cp.opt_SPDHRM_control == 1):
            veh_trajs_init = osh.opt_update_speed(veh_trajs_init, osh.congestion_location,
                                                  osh.brdcst_dist, osh.VSL_SPDHRM,
                                                  veh_active_SPDHRM)
    
    #   remove empty rows
    veh_trajs = veh_trajs[veh_trajs[:,0] >= 0 ,]
    
    #   generate travel time table
    tt_array = tm.gen_travel_time(veh_trajs)
    
    #   convert to dataframes
    veh_trajs_df = pd.DataFrame(veh_trajs,
                                columns = ['timestep', 'vehicleid', 'lane',
                                           'position', 'speed', 'gap',
                                           'delta_speed', 'accelaration',
                                           'desired_speed', 'lead_id',
                                           'lead_right_id', 'lead_left_id',
                                           'pred_id', 'pred_right_id',
                                           'pred_left_id', 'ln_chg',
                                           'maximum_decelaration', 'target_exit',
                                           'drv_bhvr', 's_star',
                                           'mon_timestep', 'section', 'vehicle_class'])
    tm_df = pd.DataFrame(tm_array,
                            columns = ['mon_timestep', 'section', 'num_vehs',
                                       'mean_speed', 'density', 'flow',
                                       'ssd', 'mean_headway','lane_changes',
                                       'acc_std', 'ssd_slead', 'mean_speed_slead',
                                       'mean_headway_slead', 'congestion',
                                       'main_volume', 'ramp_volume', 'SPDHRM', 'LNDRP'])
    
    tm_df_all = pd.DataFrame(tm_array_all,
                            columns = ['mon_timestep', 'section', 'num_vehs',
                                       'mean_speed', 'density', 'flow',
                                       'ssd', 'mean_headway','lane_changes',
                                       'acc_std', 'ssd_slead', 'mean_speed_slead',
                                       'mean_headway_slead', 'congestion',
                                       'main_volume', 'ramp_volume', 'SPDHRM', 'LNDRP'])
    tt_df = pd.DataFrame(tt_array, columns = ['veh_id', 'travel_time'])
    
    #   generate speed harmonization tracking data frame
    if(cp.SPDHRM_control == 1):
        SPDHRM_track_df = pd.DataFrame(
            {'timestep': track_timestep,
             'activeSPDHRM': track_activeSPDHRM,
             'VSL': track_VSL     
            })
    elif(cp.opt_SPDHRM_control == 1):    
        SPDHRM_track_df = pd.DataFrame(
            {'timestep': track_timestep,
             'activeSPDHRM': track_activeSPDHRM,
             'VSL': track_VSL,
             'brdcst_dist': track_brdcst_dist,
             'speed_congestion': track_spd_congestion,
             'speed_500': track_spd_500,
             'speed_1000': track_spd_1000,
             'speed_1500': track_spd_1500,
             'speed_cong_slead': track_speed_cong_slead,
             'ssd_cong_slead': track_ssd_cong_slead,
             'flow_cong': track_flow_cong,
             'density_cong': track_density_cong
            })
        
    #   Generate spatio-tempo image
    
    #vs.spatio_temp_map(tm_df,'mean_speed')
    #vs.scatter_plot(tm_df, 'density', 'flow')
    vs.spatio_temp_map(tm_df_all,'mean_speed')
    vs.scatter_plot(tm_df_all, 'density', 'flow')
    import visualization as vs
    vs.tt_hist(tt_df)
    avg_tt = np.mean(tt_df.iloc[:,1])
    print('Average Travel Time in Main Lanes = %f' %avg_tt)
    
    #   Save traffic properties
    if(cp.mac == 1):
        tm_df.to_csv('%s/IO/tm_df.csv' %module_dir, index = False)
        if((cp.dec_SPDHRM_control == 1) & (cp.dec_SPDHRM_multimodel > 0)):
            tm_df_multimodel.to_csv('%s/IO/tm_df_multimode.csv' %module_dir, index = False)
        tm_df_all.to_csv('%s/IO/tm_df_all.csv' %module_dir, index = False)
        veh_trajs_df.to_csv('%s/IO/veh_trajs.csv' %module_dir, index = False)
        tt_df.to_csv('%s/IO/tt_df.csv' %module_dir, index = False)
        if(cp.opt_SPDHRM_control == 1):
            SPDHRM_track_df.to_csv('%s/IO/SPDHRM_track_df.csv' %module_dir, index = False)
    
    else:
        tm_df.to_csv('%s\\IO\\tm_df.csv' %module_dir, index = False)
        if((cp.dec_SPDHRM_control == 1) & (cp.dec_SPDHRM_multimodel > 0)):
            tm_df_multimodel.to_csv('%s\\IO\\tm_df_multimode.csv' %module_dir, index = False)
        tm_df_all.to_csv('%s\\IO\\tm_df_all.csv' %module_dir, index = False)
        veh_trajs_df.to_csv('%s\\IO\\veh_trajs.csv' %module_dir, index = False)
        tt_df.to_csv('%s\\IO\\tt_df.csv' %module_dir, index = False)
        if(cp.opt_SPDHRM_control == 1):
            SPDHRM_track_df.to_csv('%s\\IO\\SPDHRM_track_df.csv' %module_dir, index = False)
    
#    input('press enter')
    
    
