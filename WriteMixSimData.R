require(MixSim)

ns <- c(1000, 10000, 100000)
Ks <- c(10, 40)
dims <- c(2, 10, 20)
Omegas <- c(0, 0.1)

for (n in ns){
  for (K in Ks){
    for (dim in dims){
      for (Omega in Omegas){
        for (v in 1:10){
          repeat{ 
            Q <-MixSim(MaxOmega = Omega, K = K, p = dim) 
            
            if (Q$fail == 0) break 
          } 
          
          
          Train <-simdataset(n = n, Pi = Q$Pi, Mu = Q$Mu, S = Q$S, n.out = 0, int = c(0, 1)) 
          write.table(cbind(Train$X, Train$id), paste0("Data/Simulated/Train_n", as.character(n), 
                                                     "_K", as.character(K), 
                                                     "_d", as.character(dim), 
                                                     "_Omega", as.character(Omega), 
                                                     "_v", as.character(v), 
                                                     ".csv"), 
                    row.names = F, 
                    col.names = F, 
                    sep = ",")
          
        }
      }
    }
  }
}
