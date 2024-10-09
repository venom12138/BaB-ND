export model_path=\"ckpts/pushing_T/dynamics_model.pth\"
export CUDA_VISIBLE_DEVICES="0"

python stat_planning_for_pushing_hier_real.py --config-name=pushing_T \
        planning.model_path=$model_path \
        planning.method_types=['MPPI_BF','DecentCEM','CROWN'] \
        planning.obs_pos_list=[[1.1,2.5],[3.05,2.75]] \
        +planning.test_id=0

python stat_planning_for_pushing_hier_real.py --config-name=pushing_T \
        planning.model_path=$model_path \
        planning.method_types=['MPPI_BF','DecentCEM','CROWN'] \
        planning.obs_pos_list=[[1.4,2.75],[3.9,2.75]] \
        planning.obs_size_list=[0.5,0.3] \
        +planning.test_id=1

python stat_planning_for_pushing_hier_real.py --config-name=pushing_T \
        planning.model_path=$model_path \
        planning.method_types=['MPPI_BF','DecentCEM','CROWN'] \
        planning.obs_pos_list=[[3.9,1.6],[2,3],[1,3]] \
        planning.obs_size_list=[0.4,0.4,0.4] \
        +planning.test_id=2
