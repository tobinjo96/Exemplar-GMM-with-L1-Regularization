

library(Rmixmod)
library(mclust)

ns <- c(1000, 10000, 100000)
Ks <- c(40)
dims <- c(2, 10, 20)
Omegas <- c(0, 0.1)

for (n in ns){
  for (K in Ks){
    for (dim in dims){
      for (Omega in Omegas){
        for (v in 1:10){
          
          Train <- read.table(paste0("Data/Simulated/Train_n", as.character(n), 
                                     "_K", as.character(K), 
                                     "_d", as.character(dim), 
                                     "_Omega", as.character(Omega), 
                                     "_v", as.character(v), 
                                     ".csv"), 
                              header = 0, 
                              sep = ",")
          
          X_train <- Train[, 1:dim]
          y_train <- Train[, dim + 1]
          # 
          t1 <- system.time({ 
            strategy <- mixmodStrategy(algo = "EM", initMethod = "smallEM")
            mixmodclus <- mixmodCluster(data = X_train, nbCluster = 2:(K + 10),
                                        criterion = c("BIC", "ICL", "NEC"), models = mixmodGaussianModel(),
                                        strategy = strategy)
            
          })
          
          forecasts <- mixmodclus@bestResult@partition
          
          write.table(paste0(as.character(n), ",",
                             as.character(dim), ",",
                             as.character(K), ",",
                             as.character(Omega), ",",
                             as.character(v), ",",
                             as.character(t1[3]), ",",
                             as.character(length(unique(forecasts))),",",
                             adjustedRandIndex(y_train, forecasts), ",",
                             as.character(mixmodclus@bestResult@criterionValue[1]),",",
                             as.character(mixmodclus@bestResult@criterionValue[2]),",",
                             as.character(mixmodclus@bestResult@criterionValue[3])),
                      file = "MixModemEM_Assess.csv", 
                      append = T, 
                      sep=',', 
                      row.names=F, 
                      col.names=F)
          
        }
      }
    }
  }
}

datasets <- c("dermatology", "ecoli", "optdigits", "pendigits", "seeds")
for (dataset in datasets){
  
  
  Train <- read.table(paste0("Data/Real/", 
                             dataset,
                             ".csv"), 
                      header = 0, 
                      sep = ",")
  dim <- ncol(Train) - 1
  X_train <- Train[, 1:dim]
  y_train <- Train[, dim + 1]
  K <- length(unique(y_train))
  # 
  t1 <- system.time({ 
    strategy <- mixmodStrategy(algo = "EM", initMethod = "smallEM")
    mixmodclus <- mixmodCluster(data = X_train, nbCluster = max(2,(K-5)):(K + 10),
                                criterion = c("BIC", "ICL", "NEC"), models = mixmodGaussianModel(),
                                strategy = strategy)
    
  })
  
  forecasts <- mixmodclus@bestResult@partition
  write.table(paste0(dataset, ",",
                     as.character(t1[3]), ",",
                     as.character(length(unique(forecasts))),",",
                     adjustedRandIndex(y_train, forecasts), ",",
                     as.character(mixmodclus@bestResult@criterionValue[1]),",",
                     as.character(mixmodclus@bestResult@criterionValue[2]),",",
                     as.character(mixmodclus@bestResult@criterionValue[3])),
              file = "MixModemEM_Assess.csv", 
              append = T, 
              sep=',', 
              row.names=F, 
              col.names=F)
}
