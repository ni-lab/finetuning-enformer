# 384 bins, LCL weights
# python train_and_predict.py ../prior_dataset/random_split.384_bins random_split.384_bins.LCL_weights --meta_feature_weight_fn LCL
# python train_and_predict.py ../prior_dataset/population_split.384_bins population_split.384_bins.LCL_weights --meta_feature_weight_fn LCL

# 384 bins, uniform weights
# python train_and_predict.py ../prior_dataset/random_split.384_bins random_split.384_bins.uniform_weights --meta_feature_weight_fn uniform
# python train_and_predict.py ../prior_dataset/population_split.384_bins population_split.384_bins.uniform_weights --meta_feature_weight_fn uniform

# 1Mb, LCL weights
python train_and_predict.py ../prior_dataset/random_split.1Mb random_split.1Mb.LCL_weights --meta_feature_weight_fn LCL
python train_and_predict.py ../prior_dataset/population_split.1Mb population_split.1Mb.LCL_weights --meta_feature_weight_fn LCL

# 1Mb, uniform weights
python train_and_predict.py ../prior_dataset/random_split.1Mb random_split.1Mb.uniform_weights --meta_feature_weight_fn uniform
python train_and_predict.py ../prior_dataset/population_split.1Mb population_split.1Mb.uniform_weights --meta_feature_weight_fn uniform
