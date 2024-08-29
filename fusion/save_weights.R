suppressMessages(library("optparse"))

option_list = list(
    make_option("--RData", type="character", help="wgt.RDat file to be loaded"),
    make_option("--out_prefix", type="character", help="Prefix for output files")
)

opt = parse_args(OptionParser(option_list=option_list))

if (is.null(opt$RData) || is.null(opt$out_prefix)) {
    stop("Missing required arguments")
}

load(opt$RData)

models = colnames(wgt.matrix)

# Iterate through the models and save the weights
for (name in models) {
    if (name == "lasso" || name == "enet" ) {
        keep = wgt.matrix[,name] != 0
    } else if (name == "top1") {
        keep = which.max(wgt.matrix[,name]^2)
    } else {
        keep = 1:nrow(wgt.matrix)
    }

    table = format(cbind(snps[,c(2,5,6)], wgt.matrix[,name])[keep,], digits=3)
    output_path = paste(opt$out_prefix, name, "weights.txt", sep=".")
    write.table(table, output_path, quote=F, row.names=F, col.names=F, sep='\t')
}
