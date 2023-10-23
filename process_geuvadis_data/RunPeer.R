library(peer)

counts = read.csv("rpkm/unnormalized_rpkm_processed.csv", header=TRUE, row.names=1) # [genes, samples]
counts = counts[, -c(1:3)] # remove first 3 columns
counts = t(counts) # [samples, genes]

model = PEER()
PEER_setNk(model, 10)
PEER_setAdd_mean(model, TRUE)
PEER_setPhenoMean(model, as.matrix(counts))
PEER_update(model)

factors = PEER_getX(model) # [samples, factors]
weights = PEER_getW(model) # [genes, factors]
residuals = t(PEER_getResiduals(model)) # [genes, samples]

write.csv(factors, "rpkm/peer/factors.csv")
write.csv(weights, "rpkm/peer/weights.csv")
write.csv(residuals, "rpkm/peer/residuals.csv")
