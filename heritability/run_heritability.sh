
basedir=/clusterfs/nilah/rkchung/data/geuvadis
for f in `ls $basedir/geuvadis_pheno/*.pheno`; do
    echo $f
    IFS='/_' read -r -a array <<< "$f"
    gene="${array[8]}"
    chr="${array[9]}"
    s="${array[10]}"
    e="${array[11]}"
    IFS='\.' read -r -a array <<< "$e"
    e="${array[0]}"
    echo $gene, $chr, $s, $e
    # Varaints around gene of interest
    /clusterfs/nilah/software/plink1.9/plink --bfile $basedir/geuvadis_plink/chr$chr.maf05.hwe05 --make-bed --chr $chr --from-bp $s --to-bp $e --out $basedir/geuvadis_plink/$gene

    # Create GRM using GCTA 
    /clusterfs/nilah/rkchung/tools/gcta_v1.94.0Beta_linux_kernel_3_x86_64/gcta_v1.94.0Beta_linux_kernel_3_x86_64_static --bfile $basedir/geuvadis_plink/$gene --autosome --maf 0.01 --make-grm --out $basedir/heritability/$gene --thread-num 10

    # Create GRM using GCTA 
    /clusterfs/nilah/rkchung/tools/gcta_v1.94.0Beta_linux_kernel_3_x86_64/gcta_v1.94.0Beta_linux_kernel_3_x86_64_static --grm $basedir/heritability/$gene --pheno $f --reml --out $basedir/heritability/$gene --thread-num 10

    grep "V(G)/Vp" $basedir/heritability/$gene.hsq | sed 's/$/\t'$gene'/' >> $basedir/heritability.maf05.hwe05.tsv
done
