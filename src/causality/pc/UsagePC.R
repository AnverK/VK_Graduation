#================================= A test case =================================
# install.packages("tictoc")
# install.packages("BiocManager")
# BiocManager::install(c("graph", "RBGL", "Rgraphviz"))
# install.packages("pcalg")
library(pcalg)
library(graph)
library(MASS)
library(tictoc)
library(igraph)

# read data
dataset_path <- file.path("dataset", "feed_top_ab_tests_pool_dataset_big.csv")
print("Loading dataset...")
dataset <- read.table(dataset_path, sep = ",", header = TRUE)
dataset <- dataset[, grepl(".*\\d$", names(dataset))]
# Prepare data
corrolationMatrix <- cor(dataset)
p <- ncol(dataset)
suffStat <- list(C = corrolationMatrix, n = nrow(dataset))

# pvalues <- c(0.001)
pvalues <- c(0.001, 0.01, 0.05, 0.1, 0.15, 0.2)
for (alpha in pvalues) {
  tic()
  fci
  stable_fast_fit <- fci(suffStat, indepTest = gaussCItest, p = p, skel.method = "stable.fast", alpha = alpha,
                         NAdelete = TRUE)
  print("the total time consumed by stable.fast is:")
  toc()
  # print(stable_fast_fit)
  # print(stable_fast_fit@graph)

  if (require(Rgraphviz)) {
    graph_title <- paste("big_r_graph_p_", alpha)
    jpeg(paste(graph_title, ".jpeg"))
    nAttrs <- list()
    graph <- stable_fast_fit@graph
    # z <- strsplit(packageDescription("Rgraphviz")$Description, " ")[[1]]
    # z <- z[1:numNodes(graph)]
    z <- c(0:3, 0:16)
    colors <- c(rep("red", 4), rep("green", 17))
    names(colors) <- nodes(graph)
    nAttrs$fillcolor <- colors
    names(z) <- nodes(graph)
    nAttrs$label <- z
    plot(graph, nodeAttrs = nAttrs, main = paste("pvalue=", graph_title))
    dev.off()
  }
}
