# Pipeline
1. Convert `eQTL_finemapping.significantAssociations.MAGE.v1.0.txt` from hg38 to hg19
    1. `liftover_to_hg19/create_vcf.py`: Create a VCF file from the above txt file
    2. `liftover_to_hg19/run_liftover.sh`: Run LiftoverVcf (Picard)
    3. `liftover_to_hg19/add_hg19_coords_to_finemapping_res.py`: Create a new file containing the hg19 coordinates
2. `subset_to_training_variants.py`: Subset to training variants (seen by both deep learning model and FUSION models) based on the following conditions:
    * eGene maps to a random-split or population-split gene in our dataset
    * Within 49.2 kb context size
    * MAF >= 5% in training set
    Produces `eQTL_finemapping.significantAssociations.trainingVariant.MAGE.v1.0.hg19.txt`
3. `../compare_variant_importance.ipynb`: Compare across models
