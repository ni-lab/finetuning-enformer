import pandas as pd
from genomic_utils.variant import Variant
from tqdm import tqdm

HG38_RES_PATH = "/data/yosef3/scratch/ruchir/data/MAGE/MAGE.v1.0.data/QTL_results/eQTL_results/eQTL_finemapping_results/eQTL_finemapping.significantAssociations.MAGE.v1.0.txt"
HG19_RES_PATH = "/data/yosef3/scratch/ruchir/data/MAGE/MAGE.v1.0.data/QTL_results/eQTL_results/eQTL_finemapping_results/hg19/eQTL_finemapping.significantAssociations.MAGE.v1.0.txt"
HG19_VCF_PATH = "/data/yosef3/scratch/ruchir/data/MAGE/MAGE.v1.0.data/QTL_results/eQTL_results/eQTL_finemapping_results/hg19/significantVariants.hg19.vcf"


def parse_info_field(info: str) -> dict[str, str]:
    map_ = {}
    for attr in info.split(";"):
        if "=" in attr:
            key, val = attr.split("=")
            map_[key] = val
    return map_


def create_hg38_to_hg19_variant_map() -> dict[Variant, Variant]:
    hg38_to_hg19_map = {}
    with open(HG19_VCF_PATH, "r") as f:
        for line in tqdm(f, desc="Creating hg38 to hg19 variant map"):
            if line.startswith("#"):
                continue
            row = line.strip().split("\t")
            hg19_v = Variant(row[0], row[1], row[3], row[4])

            # Create hg38 variant from info field
            info_dict = parse_info_field(row[-1])
            assert "OriginalContig" in info_dict and "OriginalStart" in info_dict
            if "OriginalAlleles" not in info_dict:
                hg38_v = Variant(
                    info_dict["OriginalContig"],
                    info_dict["OriginalStart"],
                    row[3],
                    row[4],
                )
            else:
                ref, alt = info_dict["OriginalAlleles"].split(",")
                hg38_v = Variant(
                    info_dict["OriginalContig"], info_dict["OriginalStart"], ref, alt
                )
            hg38_to_hg19_map[hg38_v] = hg19_v

    return hg38_to_hg19_map


def main():
    # Load the hg38 results
    res_df = pd.read_csv(HG38_RES_PATH, sep="\t", header=0)
    res_df["hg38_variant"] = [
        Variant(
            row["variantChrom"],
            row["variantPosition"],
            row["variantRef"],
            row["variantAlt"],
        )
        for _, row in res_df.iterrows()
    ]

    # Create mapping from hg38 variants to hg19 variants using the LiftOver results
    hg38_to_hg19_map = create_hg38_to_hg19_variant_map()

    # Map hg38 results to hg19 results
    res_df["hg19_variant"] = res_df["hg38_variant"].map(hg38_to_hg19_map)

    # Count the number of variants that were unmapped
    deduplicated_res_df = res_df.drop_duplicates(subset=["hg38_variant"])
    n_mapped = deduplicated_res_df["hg19_variant"].notnull().sum()
    n_total = deduplicated_res_df.shape[0]
    map_pct = n_mapped / n_total * 100
    print(
        f"# of variants that were mapped to hg19: {n_mapped}/{n_total} ({map_pct:.2f}%)"
    )

    # Write the hg19 results to a file
    res_df.to_csv(HG19_RES_PATH, sep="\t", index=False)


if __name__ == "__main__":
    main()
