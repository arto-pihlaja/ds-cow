# require(readr)
# require(ggplot2)
require(dplyr)
# data = readxl::read_xlsx('CowSample.xlsx')
data = read.csv(file="CowSample.csv", sep=";", dec=",")
names(data) =  gsub(' ','_',names(data))
names(data) =  gsub('\\.','_',names(data))
data = filter(data,Cow_Name!='Mirkku')

require(mgcv)
Cow = factor(data$Cow_Name)
m = gam(Production_avg~Cow+s(DIM,by=Cow),data=data)
new_data=data.frame(DIM=rep(0:100,nlevels(Cow)),Cow=rep(levels(Cow),each=101))
new_data$pred = predict(m,new_data)
group_by(new_data,Cow) %>% summarize(`Estimated peak (DIM)` = which.max(pred))

# For full dataset
# Sort all data by Cow
# Make one df per cow, and for each df
#   Build gam and make prediction 
#   Discard gam

cowCodes = unique(data$CowKEY)
for ck in cowCodes:
  

