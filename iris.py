import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
#******DATA IMPORT******
data_im=pd.read_csv("internshipColl/assignment_09/dataset/Iris.csv")
#*************DATA EXPLORATION*************
#print first 5 rows of sample data
print(data_im.head())
##print last 5 rows of sample data
print(data_im.tail())
#print random n number of rows in sample data
print(data_im.sample(16))
#to display total number of rows and columns in sample data
print(data_im.shape) #(rows=150,columns=6)
#to get basic staistics of each column of sample data
print(data_im.describe())
#To get information about various data types used and the non-null count of each column
print(data_im.info())
#To get memory consumtion detailes by each column
print(data_im.memory_usage())
#*************DATA SELECTION*************
#to get perticular row ***data_im.iloc[row_num]***
print(data_im.iloc[6])
#to get perticular column ***data_im["column_name"]***
print(data_im["Species"])
#to get perticular multiple columns ***data_im["column_name1",["column_name2"]]***
print(data_im[["Species","SepalLengthCm"]])
#************DATA CLEANIMG*****************
#to identify the missing values in dataframe
print(data_im.isnull())
#EX:null
print(data_im.iloc[7])
#to fill missing values in column **data_im["column_name"].fillna(value)**
print(data_im.describe())
#"inplace=True" make  changes in original dataframe(it wil not return new dataframe)
data_im["SepalLengthCm"].fillna(5.854730,inplace=True)
data_im["SepalWidthCm"].fillna(3.048322,inplace=True)
data_im["PetalLengthCm"].fillna(3.773826,inplace=True)
data_im["PetalWidthCm"].fillna(1.198667,inplace=True)
print(data_im.info())
#EX:null filed
print(data_im.iloc[7])
#to remove missing values row **data_im.dropna()**
data_im.dropna(inplace=True)
print(data_im.info())
#to convert the data type of the selected columns to a different data type 
#"data_im["column_name"].astype(new_data_type)"
#ex:data_im[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].astype(float)
#as they are already in float format we don't require this right now

#*****************DATA ANALYSIS*********************
#grouping,sorting,filtering
print(data_im["Species"].value_counts())

#To group a column by its name and then apply some aggregation function 
#like sum,mean,max/min etc
p=data_im.groupby("Species").agg({"PetalWidthCm":"sum"})
print(p)

#*****filtering data***
#df[df["sales"]>5000] // will display rows where the value of sales is greater than 5000
#df.query("SALES">5000) //we can also filter the data frame using query


#*****sorting data***
# to sort specific column, either in ascending/descending order
#df.sort_values("SALES",ascending=False) //sort in descending order
#pivot table
#pd.pivot_table(df,values="SALES",index="CITY",columns="YEAR_ID",aggfunc="sum")
#pivot tables summarizeata using specific columns.
# this is very useful in analyzing the data when you only want to consider
#  the effect of particular columns
"""
values:it contains the column for which you want to populate the table's cells
index:The column used in it will become the row index of the pivot table,and each unique 
      category of this column will become a row in the pivot table.
columns:it contains the headers of the pivot table, and each unique element
        will become the column in the pivot table
aggfunc:This is the same aggregator function we doscussed earlier
"""

#ploting using pandas
data_im.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm",color="green",s=7)
plt.show()

#ploting using seaborn to make bivariate scatterplots
#  and univariate histograms
sea.jointplot(x="SepalLengthCm",y="SepalWidthCm",data=data_im,size=5)
plt.show()

#modifing above graph with each column eith different color
sea.FacetGrid(data_im,hue="Species") \
.map(plt.scatter,"SepalLengthCm","SepalWidthCm")\
.add_legend()
plt.show()


#change of data points by assigning a list of colors
KS={"color":["red","green","yellow"]}
sea.FacetGrid(data_im,hue_kws=KS,hue="Species") \
.map(plt.scatter,"SepalLengthCm","SepalWidthCm")\
.add_legend()
plt.show()

#To plot box plot of species
sea.boxplot(x="Species",y="PetalLengthCm",data=data_im)
plt.show()

#Use seaborn's striplot to add points on top of the box plot
#jitter=True data points remain scattered and piled into a verticle line.
#assign ax to each axis, so that each plot is ontop of previous axis.
ax=sea.boxplot(x="Species",y="PetalLengthCm",data=data_im)
ax=sea.stripplot(x="Species",y="PetalLengthCm",data=data_im,jitter=True,edgecolor="gray")
plt.show()


#Tweek the plot above to change fill and border color using ax.artists.
#assing ax.artists a variable name, and insert the box number into the corresponding brackets

"""
ax=sea.boxplot(x="Species",y="PetalLengthCm",data=data_im)
ax=sea.stripplot(x="Species",y="PetalLengthCm",data=data_im,jitter=True,edgecolor="gray")
boxtwo=ax.artists[1]
boxtwo.set_facecolor('red')
boxtwo.set_edgecolor('black')
boxthree=ax.artists[0]
boxthree.set_facecolor('yellow')
boxthree.set_edgecolor('black')
plt.show()
"""
#violin plot-density of data , similarly to a scatter plot,
#and presents catagorical data like a box plot.
#Denser regions of the data are fatter.
sea.violinplot(x="Species",y="PetalLengthCm",data=data_im)
plt.show()

#To change the fill color of the violin,choose desired colors and set equal to pallete
sea.violinplot(x="Species",y="PetalLengthCm",palette={"blue","red","yellow"},data=data_im)
plt.show()

#seaborn's kdeplot, plots univariate or bivariate density estimates.
#size can be changed by tweeking the value used
sea.FacetGrid(data_im,hue="Species")\
.map(sea.kdeplot,"PetalLengthCm")\
.add_legend()
plt.show()

#use pairplot to analyze the relatinship between species for
#all characteristic combinations.
#an observable trend shows a close relationship between two
#os the species
sea.pairplot(data_im.drop("Id",axis=1),hue="Species")
plt.show()


#set diag_kind equal to kde to modify diagnal elements into showing kernal
#density estimation.
sea.pairplot(data_im.drop("Id",axis=1),hue="Species",diag_kind="kde")
plt.show()

#use seaborn's jointplot to make a hexagonal binplot
#set desired size and ratio and choose a color.
sea.jointplot(x="SepalLengthCm",y="SepalWidthCm",data=data_im,ratio=10,kind='hex',color='pink')
plt.show()

#To make a pandas boxplot grouped by species,use .boxplot
#modify the figsize,by placing a value in the X and Y cordinates
data_im.drop("Id",axis=1).boxplot(by="Species",figsize=(10,110))
plt.show()

#In pandas use Andrews Curves to plot and visualize data structure.
#Each multivariate observation is transformed into a curve 
# and represents the coefficients of a Fc
#This useful for detecting outliers in times series data.
#use colormap to change to change the color of the curves

pd.plotting.andrews_curves(data_im("Id",axis=1),"Species",colormap='rainbow')
plt.show()

#use pandas


pd.plotting.andrews_curves(data_im("Id",axis=1),"Species",colormap='rainbow')
plt.show()



