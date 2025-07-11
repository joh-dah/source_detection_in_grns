if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("OmnipathR","dorothea", "decoupleR"), ask = FALSE)

library(dorothea)
library(decoupleR)

# Load all human DoRothEA regulons (levels A–E)
regulons <- get_dorothea(organism = "human", levels = c("A", "B", "C", "D", "E"))

# Export as CSV
write.csv(regulons, "dorothea_human_full.csv", row.names = FALSE)
