
 library(tidyverse)
 
 library(factoextra)
 library(cluster)
 
 
 df = read.csv(file.choose(), h=T)
 
 names(df)

 # Removendo outliers > 3
 
 consulta_1 = df %>% filter(purchase_value < 3)  %>%  select(producer_id, 
                                                             purchase_value,product_niche, ) 
 
 consulta_1
 
 summary(consulta_1) 
 
 boxplot(consulta_1)
 
 hist(consulta_1$purchase_value)

 
 q1 <- quantile(consulta_1$purchase_value, 0.2)
 q2 <- quantile(consulta_1$purchase_value, 0.4)
 q3 <- quantile(consulta_1$purchase_value, 0.6)
 q4 <- quantile(consulta_1$purchase_value, 0.8)
 q4 <- quantile(consulta_1$purchase_value, 0.8)
 
 #consulta_1$score <- rep(NA, length(consulta_1$purchase_value))
 
 consulta_1$score <-  case_when(consulta_1$purchase_value <= q1 ~ 1,
             consulta_1$purchase_value > q1 & consulta_1$purchase_value <= q2  ~ 2,
             consulta_1$purchase_value > q2 & consulta_1$purchase_value <= q3  ~ 3,
             consulta_1$purchase_value > q3 & consulta_1$purchase_value <= q4  ~ 4,
             consulta_1$purchase_value > q4  ~ 5 )

 summary(consulta_1$score) 
 
 
 obj = consulta_1 %>% group_by(producer_id) %>% summarise(Errado = n())
 

 a= head(obj[order(obj$Errado, decreasing = T),], 30)
 
 a
 
 dat = consulta_1 %>%  filter(producer_id %in% a$producer_id)
 
 consulta_1
 
 dat

 data = anti_join(consulta_1,dat,  by = c('producer_id' = 'producer_id') )
 
 data
 
 (nrow(data) + nrow(dat)) == nrow(consulta_1)
 
 par(mfrow=c(1,2))
 
 barplot(table(dat$score))
 
 barplot(table(data$score))

 total = sum(consulta_1$score)
 
 # top 20 produtores
 top_20 = sum(dat$score)
 
 (top_20/total)*100
 
 
 names(consulta_1)
 
   
 x <- consulta_1 %>% select(purchase_value, product_niche)
 
 x
 
 table(consulta_1$product_niche)
 
 mod = kmeans(x, 3)

 
 fviz_nbclust(consulta_1, kmeans, method = "wss")   
  
 
 library(caret)
 
 caret::dummyVars(consulta_1$product_niche)
 
 dummy <- dummyVars(" ~ .", data=consulta_1 %>% select(product_niche ))

 final_df <- data.frame(predict(dummy, newdata=consulta_1 %>% select(product_niche )))

 t = final_df 
 
 
 dez = cbind(x, final_df)
 
 mod = kmeans(dez, 3)
 
 
 fviz_nbclust(dez, kmeans, method = "wss")  
 