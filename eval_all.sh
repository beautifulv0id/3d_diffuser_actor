#./online_evaluation_rlbench/eval_pointattn_nf.sh
#./online_evaluation_rlbench/eval_pointattn_nf1.sh
#./online_evaluation_rlbench/eval_pointattn_nf_1.sh
#./online_evaluation_rlbench/eval_pointattn_nf1_1.sh
#./online_evaluation_rlbench/eval_pointattn_nf_2.sh
#./online_evaluation_rlbench/eval_pointattn_nf1_2.sh
#./online_evaluation_rlbench/eval_pointattn_nf2_2.sh
#./online_evaluation_rlbench/eval_pointattn_nf2_1.sh

num_eval_iters=4
for ((j=1; j<num_eval_iters; j++)); do
#  ./online_evaluation_rlbench/eval_pointattn_URSA_final_1.sh $j
#  ./online_evaluation_rlbench/eval_pointattn_3DDA_final_1.sh $j
#  ./online_evaluation_rlbench/eval_pointattn_URSA_final_2.sh $j
#  ./online_evaluation_rlbench/eval_pointattn_3DDA_final_2.sh $j
#  ./online_evaluation_rlbench/eval_pointattn_URSA_final_3.sh $j
#  ./online_evaluation_rlbench/eval_pointattn_3DDA_final_3.sh $j

#  ./online_evaluation_rlbench/eval_pointattn_URSA_final_4.sh $j
#  ./online_evaluation_rlbench/eval_pointattn_3DDA_final_4.sh $j

  ./online_evaluation_rlbench/eval_pointattn_URSA_final_5.sh $j
#  ./online_evaluation_rlbench/eval_pointattn_3DDA_final_5.sh $j


done