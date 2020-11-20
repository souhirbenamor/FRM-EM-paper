
#[1] Import optimal weights results from 5 approaches/ EMs
df1= pd.read_excel("optimal weights according to the 5 approaches.xlsx")
print(df)

#[2]
# library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# Dataset
df1 = pd.DataFrame(df1.set_index("Date") , columns=['India (IS)',	'Brasil (BS)',	'Mexico (MF)',	'Russia (RM)',	'Turkey (TI)',	'South Africa (SJ)'])
 
# stacked barplot
#df.plot.area() # area plot
df1.plot.bar(stacked=True, figsize=(15,5)) #stacked bar plot
#df.plot( figsize=(15,5)) #Multivariate time series
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("optimal weights according to the 5 approaches.png",dpi=300, bbox_inches='tight',transparent=True)
