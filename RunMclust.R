library(mclust)

ns <- c(1000, 10000, 100000)
Ks <- c(40)
dims <- c(2, 10)
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
            mcl <- mclustBIC(X_train, G = 2:(K+10), modelNames = "VVV")
            Mclus <- Mclust(X_train, x = mcl)
          })
          
          forecasts <- Mclus$classification
          
          write.table(paste0(as.character(n), ",",
                             as.character(dim), ",",
                             as.character(K), ",",
                             as.character(Omega), ",",
                             as.character(v), ",",
                             as.character(t1[3]), ",",
                             as.character(length(unique(forecasts))),",",
                             adjustedRandIndex(y_train, forecasts), ",",
                             as.character(Mclus$bic),",",
                             as.character(Mclus$icl)),
                      file = "Mclust_Assess.csv", 
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




#Note: Mclust can fail for higher-dimensional datasets due to the production of singularities or covariances without full rank. 
#To counteract this, covariance matrices may be restricted in shape, or jitter added to the data to reduce this risk. 

Train <- read.table(paste0("Data/Real/", 
                           'dermatology',
                           ".csv"), 
                    header = 0, 
                    sep = ",")
dim <- ncol(Train) - 1
X_train <- Train[, 1:dim]
y_train <- Train[, dim + 1]
K <- length(unique(y_train))

set.seed(26)
t1 <- system.time({ 
  mcl <- mclustBIC(X_train, G = 2:(K+10), modelNames = "VVV")
  Mclus <- Mclust(X_train, x = mcl)
})

forecasts <- Mclus$classification

write.table(paste0('dermatology', ",",
                   as.character(t1[3]), ",",
                   as.character(length(unique(forecasts))),",",
                   adjustedRandIndex(y_train, forecasts), ",",
                   as.character(Mclus$bic),",",
                   as.character(Mclus$icl)),
            file = "Mclust_Assess.csv", 
            append = T, 
            sep=',', 
            row.names=F, 
            col.names=F)



Train <- read.table(paste0("Data/Real/", 
                           'ecoli',
                           ".csv"), 
                    header = 0, 
                    sep = ",")
dim <- ncol(Train) - 1
X_train <- Train[, 1:dim]
y_train <- Train[, dim + 1]
K <- length(unique(y_train))

t1 <- system.time({ 
  mcl <- mclustBIC(X_train, G = 2:(K+10), modelNames = "VVV")
  Mclus <- Mclust(X_train, x = mcl)
})

forecasts <- Mclus$classification

write.table(paste0('ecoli', ",",
                   as.character(t1[3]), ",",
                   as.character(length(unique(forecasts))),",",
                   adjustedRandIndex(y_train, forecasts), ",",
                   as.character(Mclus$bic),",",
                   as.character(Mclus$icl)),
            file = "Mclust_Assess.csv", 
            append = T, 
            sep=',', 
            row.names=F, 
            col.names=F)




Train <- read.table(paste0("Data/Real/", 
                           'optdigits',
                           ".csv"), 
                    header = 0, 
                    sep = ",")
dim <- ncol(Train) - 1
X_train <- Train[, 1:dim]
y_train <- Train[, dim + 1]
K <- length(unique(y_train))

t1 <- system.time({ 
  mcl <- mclustBIC(X_train, G = 2:(K+10), modelNames = "VVV")
  Mclus <- Mclust(X_train, x = mcl)
})

forecasts <- Mclus$classification

write.table(paste0('optdigits', ",",
                   as.character(t1[3]), ",",
                   as.character(length(unique(forecasts))),",",
                   adjustedRandIndex(y_train, forecasts), ",",
                   as.character(Mclus$bic),",",
                   as.character(Mclus$icl)),
            file = "Mclust_Assess.csv", 
            append = T, 
            sep=',', 
            row.names=F, 
            col.names=F)

Train <- read.table(paste0("Data/Real/", 
                           'pendigits',
                           ".csv"), 
                    header = 0, 
                    sep = ",")
dim <- ncol(Train) - 1
X_train <- Train[, 1:dim]
y_train <- Train[, dim + 1]
K <- length(unique(y_train))

t1 <- system.time({ 
  mcl <- mclustBIC(X_train, G = 2:(K+10), modelNames = "VVV")
  Mclus <- Mclust(X_train, x = mcl)
})

forecasts <- Mclus$classification

write.table(paste0('pendigits', ",",
                   as.character(t1[3]), ",",
                   as.character(length(unique(forecasts))),",",
                   adjustedRandIndex(y_train, forecasts), ",",
                   as.character(Mclus$bic),",",
                   as.character(Mclus$icl)),
            file = "Mclust_Assess.csv", 
            append = T, 
            sep=',', 
            row.names=F, 
            col.names=F)







Train <- read.table(paste0("Data/Real/", 
                           'seeds',
                           ".csv"), 
                    header = 0, 
                    sep = ",")
dim <- ncol(Train) - 1
X_train <- Train[, 1:dim]
y_train <- Train[, dim + 1]
K <- length(unique(y_train))
t1 <- system.time({ 
  mcl <- mclustBIC(X_train, G = 2:(K+10), modelNames = "VVV")
  Mclus <- Mclust(X_train, x = mcl)
})

forecasts <- Mclus$classification

write.table(paste0('seeds', ",",
                   as.character(t1[3]), ",",
                   as.character(length(unique(forecasts))),",",
                   adjustedRandIndex(y_train, forecasts), ",",
                   as.character(Mclus$bic),",",
                   as.character(Mclus$icl)),
            file = "Mclust_Assess.csv", 
            append = T, 
            sep=',', 
            row.names=F, 
            col.names=F)





