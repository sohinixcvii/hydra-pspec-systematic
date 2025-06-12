'''Flags'''
dummy_data_flag = False
systematics_data_flag = 'ones' # 'ones','dummy' or 'sim'
noise_flag = True


Niter = 10

'''Visibility Paths'''
parent_dir = '/Users/user/Documents/Codes/hydra_sys_project1/hydra-pspec-systematic-multiplicative/'
clean_vis_path = parent_dir+'/test_data/vis-eor-fgs.uvh5'
eor_vis_path = parent_dir+'/test_data/vis-eor.uvh5'
fg_vis_path = parent_dir+'/test_data/vis-ptsrc-gsm.uvh5'
# corrupted_vis_path = parent_dir+'/hera_val/vis_corrupted_test.uvh5'
corrupted_vis_path = parent_dir+'vis_corr_select_modes.uvh5'

op_dir = parent_dir+'outputs/debug_plots/2_modes_bsysones_clean_vis/sys_solver_tests/'
# result_dir='/Users/user/Documents/Codes/hydra_sys_project1/Results_multiplicative/results-seed-7123689-Niter-10_60_61_62_DL_modes/0-1/'
result_dir = '/Users/user/Documents/Codes/hydra_sys_project1/Results_multiplicative/results-seed-7123689-Niter-10_residuals_2dl_mode/0-1/'
# result_dir = '/Users/user/Documents/Codes/hydra_sys_project1/Results_multiplicative/results-seed-7123689-Niter-10_residuals_all_modes/0-1/'
'''Matrix file paths'''
eor_cov_path = parent_dir+'/test_data/eor-cov.npy'
N_cov_path = parent_dir+'/test_data/noise-cov.npy'
noise_path = parent_dir+'/test_data/noise.npy'
fgmodes_path = parent_dir+'/test_data/fgmodes.npy'

'''Output paths'''
if dummy_data_flag:
    if noise_flag:
        output_dir_path = 'outputs/'+'dummy_data_noise/'+'systematics_'+systematics_data_flag+'/'
    else:
        output_dir_path = 'outputs/'+'dummy_data_no_noise/'+'systematics_'+systematics_data_flag+'/'
else:
    if noise_flag:
        output_dir_path = 'outputs/'+'airy_beam_noise/'+'systematics_'+systematics_data_flag+'/'
    else:
        output_dir_path = 'outputs/'+'airy_beam_no_noise/'+'systematics_'+systematics_data_flag+'/'


'''Other variables'''
num_fg_modes = 12
nm_list=[[0, 0], [49, 0],] # [98, 0], [147, 0], [196, 0], [245, 0], [295, 0]]
