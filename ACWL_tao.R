


library(rpart)
require(nnet)



######### functions for ACWL and simulations ########

# function to sample treatment A
# input matrix.pi as a matrix of sampling probabilities, which could be non-normalized
A.sim<-function(matrix.pi){
  N<-nrow(matrix.pi) # sample size
  K<-ncol(matrix.pi) # treatment options
  if(N<=1 | K<=1) stop("Sample size or treatment options are insufficient!")
  if(min(matrix.pi)<0) stop("Treatment probabilities should not be negative!")

  # normalize probabilities to add up to 1 and simulate treatment A for each row
  pis<-apply(matrix.pi,1,sum)
  probs<-matrix(NA,N,K)
  A<-rep(NA,N)
  for(i in 1:N){
    probs[i,]<-matrix.pi[i,]/pis[i]
    A[i]<-sample(0:(K-1),1,prob=probs[i,])
  }
  class.A<-sort(unique(A))
  colnames(probs)<-paste("A=",class.A,sep="")
  return(list(A = A, probs = probs))
}

# function to estimate propensity score
# input treatment vector A and covariate matrix Xs
M.propen<-function(A,Xs){
  if(ncol(as.matrix(A))!=1) stop("Cannot handle multiple stages of treatments together!")
  if(length(A)!= nrow(as.matrix(Xs))) stop("A and Xs do not match in dimension!")
  if(length(unique(A))<=1) stop("Treament options are insufficient!")
  class.A<-sort(unique(A))

  # require(nnet)
  s.data<-data.frame(A,Xs)
  # multinomial regression with output suppressed
  model<-capture.output(mlogit<-multinom(A ~., data=s.data))
  s.p<-predict(mlogit,s.data,"probs")
  colnames(s.p)<-paste("A=",class.A,sep="")

  s.p
}

# function to estimate conditional means for multiple stages
# input Y as a continous outcome of interest
# input As = (A1, A2, ...) as a matrix of treatments at multiple stages; Stage t has treatment K_t options labeled as 0, 1, ..., K_t-1.
# input H as a matrix of covariates before assigning final treatment, excluding previous treatment variables
Reg.mu<-function(Y,As,H){
  if(nrow(as.matrix(As))!=nrow(as.matrix(H))) stop("Treatment and Covariates do not match in dimension!")
  Ts<-ncol(as.matrix(As)) # number of stages
    
  N<-nrow(as.matrix(As))
  if(Ts<0 || Ts>3) stop("Only support 1 to 3 stages!")
  H<-as.matrix(H)

  if(Ts==1){
    A1<-as.matrix(As)[,1]
    KT<-length(unique(A1)) # treatment options at last stage
    if(KT<2) stop("No multiple treatment options!")

    RegModel<-lm(Y ~ H*factor(A1))

    mus.reg<-matrix(NA,N,KT)
    for(k in 1:KT) mus.reg[,k]<-predict(RegModel,newdata=data.frame(H,A1=rep(sort(unique(A1))[k],N)))
  }
  if(Ts==2){
    A1<-as.matrix(As)[,1];A2<-as.matrix(As)[,2]
    KT<-length(unique(A2))
    if(KT<2) stop("No multiple treatment options!")

    RegModel<-lm(Y ~ (H + factor(A1))*factor(A2))

    mus.reg<-matrix(NA,N,KT)
    for(k in 1:KT) mus.reg[,k]<-predict(RegModel,newdata=data.frame(H,A1,A2=rep(sort(unique(A2))[k],N)))
  }
  if(Ts==3){
    A1<-as.matrix(As)[,1];A2<-as.matrix(As)[,2];A3<-as.matrix(As)[,3]
    KT<-length(unique(A3))
    if(KT<2) stop("No multiple treatment options!")

    RegModel<-lm(Y ~ (H + factor(A1) + factor(A2))*factor(A3))

    mus.reg<-matrix(NA,N,KT)
    for(k in 1:KT) mus.reg[,k]<-predict(RegModel,newdata=data.frame(H,A1,A2,A3=rep(sort(unique(A3))[k],N)))
  }

  output<-list(mus.reg, RegModel)
  names(output)<-c("mus.reg","RegModel")
  output
}

# function to calculate AIPW adaptive contrasts and working orders
# input outcome Y, treatment vector A, estimated propensity matrix pis.hat and regression-based conditional means mus.reg
CL.AIPW<-function(Y,A,pis.hat,mus.reg){
  class.A<-sort(unique(A))
  K<-length(class.A)
  N<-length(A)
  if(K<2 | N<2) stop("No multiple treatments or samples!")
  if(ncol(pis.hat)!=K | ncol(mus.reg)!=K | nrow(pis.hat)!=N | nrow(mus.reg)!=N) stop("Treatment, propensity or conditional means do not match!")

  # cat("pis.hat: ------------------>>>>>>>>>>>>>>>> ", pis.hat, "\n\n\n")

  #AIPW estimates; this is bias correction step
  mus.a<-matrix(NA,N,K)
  for(k in 1:K){
    mus.a[,k]<-(A==class.A[k])*Y/pis.hat[,k]+(1-(A==class.A[k])/pis.hat[,k])*mus.reg[,k]
  }


  mus.test<-matrix(NA,N,K)
  for(k in 1:K){
    mus.test[,k]<-(A==class.A[k])*Y
  }

  # cat("pis.hat: ------------------>>>>>>>>>>>>>>>> ", pis.hat, "\n\n\n")


  # C.a1 and C.a2 are AIPW contrasts; l.a is AIPW working order
  C.a1<-C.a2<-l.a<-rep(NA,N)
  for(i in 1:N){
    # largest vs. second largest
    C.a1[i]<-max(mus.a[i,])-sort(mus.a[i,],decreasing=T)[2]
    # largest vs. smallest
    C.a2[i]<-max(mus.a[i,])-min(mus.a[i,])
    # minus 1 to match A's range of 0,...,K-1
    l.a[i]<-which(mus.a[i,]==max(mus.a[i,])) # -1
  }
  output<-data.frame(C.a1, C.a2, l.a)
  output
}

# function to summarize simulation results
summary2<-function(x){
  s1<-summary(x)
  if(is.matrix(x)){
    SD<-apply(x,2,sd)
  } else{
    SD<-sd(x)
  }
  s2<-list(s1,SD)
  names(s2)<-c("summary","SD")
  return(s2)
}


ensure_vector <- function(var) {
  if (is.null(var)) {
    stop("Error: Variable is NULL")
  }
  if (!is.vector(var) || dim(var) != NULL) {
    var <- as.vector(var)  # Convert to vector if it's not
  }
  return(var)
}



# Work with  actions 1, 2, 3,  
train_ACWL <- function(job_id, S1, S2, A1, A2, probs1, probs2, R1, R2, config_number, contrast = 1) {

  cat("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<----------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

  cat("\n")
  cat("Train model: tao")
  cat("\n")

  N <- nrow(S1) 
  cat("Number of row in O1 is: ", N, "\n ")

  cat("Number of row in S1, R1, S2 is: ", class(S1), dim(S1), class(R1), dim(R1), class(S2), dim(S2), "\n ")

  # cat("Debug: Dimensions of train_input_np are", dim(O1), "and the data type is", class(O1), "\n")

  A1 <- ensure_vector(A1)
  A2 <- ensure_vector(A2)
  R1 <- ensure_vector(R1)
  R2 <- ensure_vector(R2)

  colnames_S1 = paste("x1", 1:ncol(S1), sep="")

  #################################### STAGE 2 ESTIMATION ##########################################
  # Stage 2 Estimation (Backward induction) : estimate conditional means by regression

  ####### ACWL-Contrast #########
    # Weighted classification using CART
  # if (!is.matrix(S2)) {
  if (is.matrix(S2) && (nrow(S2) == 0 || ncol(S2) == 0)) {
    # Handle the case where S2 is empty
    REG2.a1 <- Reg.mu(Y = R2, As = cbind(A1, A2), H = cbind(S1, R1))
  } else {
    # Regular case where S2 has data      
    REG2.a1 <- Reg.mu(Y = R2, As = cbind(A1, A2), H = cbind(S1, R1, S2))
  }

  mus2.reg.a1 <- REG2.a1$mus.reg
  CLs2.a1 <- CL.AIPW(Y = R2, A = A2, pis.hat = probs2, mus.reg = mus2.reg.a1)

  # main difference betwen two contrasts here!!!  
  C2.a1<-CLs2.a1$C.a1 
  if(contrast == 2){
      C2.a1<-CLs2.a1$C.a2
  }
  l2.a1<-CLs2.a1$l.a
    
  # Weighted classification using CART
  # if (!is.matrix(S2)) {
  if (is.matrix(S2) && (nrow(S2) == 0 || ncol(S2) == 0)) {
    # Handle the case where S2 is empty
    train_input_S2 <- setNames(data.frame(S1, A1, R1), c(colnames_S1, "A1", "R1"))
  } else {
    # Regular case where S2 has data      
    colnames_S2= paste("x2", 1:ncol(S2), sep="") 
    train_input_S2 <- setNames(data.frame(S1, A1, R1, S2), c(colnames_S1, "A1", "R1", colnames_S2))
  }     
  fit2.a1 <- rpart(l2.a1 ~ ., data = train_input_S2, weights = C2.a1, method = "class")
  g2.a1 <- as.numeric(predict(fit2.a1, type = "class")) 
    


  #################################### STAGE 1 ESTIMATION ##########################################
  E.R2.a1 <- rep(NA, N)
  for (m in 1:N){
      E.R2.a1[m] <- R2[m] + mus2.reg.a1[m, g2.a1[m]] - mus2.reg.a1[m, A2[m]]
  }
    
  # calculate pseudo outcome (PO)
  PO.a1 <- R1 + E.R2.a1

  ####### ACWL-Contrast #########
  REG1.a1 <- Reg.mu(Y = PO.a1, As = A1, H = S1)
  mus1.reg.a1 <- REG1.a1$mus.reg
  CLs1.a1 <- CL.AIPW(Y = PO.a1, A = A1, pis.hat = probs1, mus.reg = mus1.reg.a1)

  # main difference betwen two contrasts here!!!  
  C1.a1<-CLs1.a1$C.a1
  if(contrast == 2){
      C1.a1<-CLs1.a1$C.a2
  }
  l1.a1<-CLs1.a1$l.a

  # Weighted classification using CART
  train_input_S1 <- setNames(as.data.frame(S1), colnames_S1)
  fit1.a1 <- rpart(l1.a1 ~ ., data = train_input_S1, weights = C1.a1, method = "class")
  g1.a1 <- as.numeric(predict(fit1.a1, type = "class")) 
    

  cat("Saving Tao's model now \n")
  # Check if the directory exists, if not, create it, including any necessary parent directories
  model_dir <- sprintf("models/%s", job_id)
  if (!dir.exists(model_dir)) {
      dir.create(model_dir, recursive = TRUE)
      if (dir.exists(model_dir)) {
          cat(sprintf("Directory '%s' created successfully.\n", model_dir))
      } else {
          stop(sprintf("Failed to create directory '%s'.", model_dir))
      }
  } else {
      cat(sprintf("Directory '%s' already exists.\n", model_dir))
  }

  
  # Define and save the file paths for the second set of models
  file_path_fit1 <- sprintf("%s/best_model_ACWL_stage1_tao_fit1_config_number_%d.rds", model_dir, config_number)
  file_path_fit2 <- sprintf("%s/best_model_ACWL_stage2_tao_fit2_config_number_%d.rds", model_dir, config_number)
  saveRDS(fit1.a1, file = file_path_fit1)
  saveRDS(fit2.a1, file = file_path_fit2)

}




test_ACWL <- function(S1, S2, A1, A2, R1, R2, config_number, job_id) {

  cat("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<----------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

  cat("\n")
  cat("Test model: tao")
  cat("\n")

  ni <- nrow(S1) 
  cat("Number of rows in O1 is: ", ni, "\n")

  # cat("S1: ", class(S1), dim(S1), "\n") # S1 is matrix type
  # cat("S2: ", class(S2), dim(S2), "\n") # S2 is matrix type

  A1 <- ensure_vector(A1)
  A2 <- ensure_vector(A2)

  R1 <- ensure_vector(R1)
  R2 <- ensure_vector(R2)

  cat("Number of row in S1, R1, S2 is: ", class(S1), dim(S1), class(R1), dim(R1), class(S2), dim(S2), "\n ")


  E0.yi <- matrix(NA, ni, 2)  # a matrix of estimated counterfactual mean by different methods
 
  # Initializing vectors for storage
  g1.a1 <- g2.a1 <- numeric(ni)

  model_dir <- sprintf("models/%s", job_id)

  # Loading saved models with config number in the file names
  fit1.a1 <- readRDS(sprintf("%s/best_model_ACWL_stage1_tao_fit1_config_number_%d.rds", model_dir, config_number))
  fit2.a1 <- readRDS(sprintf("%s/best_model_ACWL_stage2_tao_fit2_config_number_%d.rds", model_dir, config_number))
        

  colnames_S1 = paste("x1", 1:ncol(S1), sep="") 

  # Predicting the optimal g and plug in the true outcome model for counterfactual mean
  for (k in 1:ni) {

    ################################    ACWL  ################################

    newdata1.a1 <- data.frame(t(S1[k, ] ))
    colnames(newdata1.a1) <- colnames_S1 
  
    g1.a1[k] <- as.numeric(predict(fit1.a1, newdata = newdata1.a1, type = "class")) 

    # Regular case where S2 has data  
    colnames_S2= paste("x2", 1:ncol(S2), sep="") 
    # print("colnames_S2: ", colnames_S2)
    newdata2.a1 <- data.frame(S1[k, , drop = FALSE], A1[k], R1[k], S2[k, , drop = FALSE])     
    colnames(newdata2.a1) <- c(colnames_S1, "A1", "R1", colnames_S2)

    # Debug: Print the columns of newdata2.a1 to check if 'x21' is present
    # print(colnames(newdata2.a1))

    g2.a1[k] <- as.numeric(predict(fit2.a1, newdata = newdata2.a1, type = "class")) 

  }

  return(list(
    g1.a1 = g1.a1,
    g2.a1 = g2.a1
  ))
}










