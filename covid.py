import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
datas=pd.read_csv("internshipColl/assignment_09/dataset/Covid Data.csv")
#print random n number of rows in sample data
print(datas.head())
print(datas.shape)
#replacing 97 & 99 blank as these are unknow
datas.replace(97,0,inplace=True)
datas.replace(99,0,inplace=True)
print(datas.head())
print(datas.info())
#***************Basic Level Questions:*********************
#1 What is the distribution of COVID-19 cases by sex?
sex_counts=datas["SEX"].value_counts()
print(sex_counts)
gender=sex_counts.index
total_gender=sex_counts.values
print(gender)
print(total_gender)
plt.bar(gender,total_gender)
plt.xlabel("sex of patients")
plt.ylabel("number of patients")
plt.show()
#conclustion:females are highly infected as compare to male



#2 How does age distribution vary among COVID-19 positive patients?
age_counts=datas["AGE"].value_counts()
print(age_counts)
age_index=age_counts.index
age_value=age_counts.values
plt.bar(age_index,age_value)
plt.xlabel("age of patients")
plt.ylabel("number of patients")
plt.show()
#conclustion:range of 24-29 are infected more.majority of them are of 28 years old

#3 What is the percentage of patients with pre-existing conditions like diabetes,
# hypertension, etc., among all COVID-19 cases?
diabetes=(datas["DIABETES"]==1).value_counts()
hipertension=(datas["HIPERTENSION"]==1).value_counts()
pregnent=(datas["PREGNANT"]==1).value_counts()
copd=(datas["COPD"]==1).value_counts()
asthma=(datas["ASTHMA"]==1).value_counts()
inmsupr=(datas["INMSUPR"]==1).value_counts()
pneumonia=(datas["PNEUMONIA"]==1).value_counts()
cardiovascular=(datas["CARDIOVASCULAR"]==1).value_counts()
obesity=(datas["OBESITY"]==1).value_counts()
tobacco=(datas["TOBACCO"]==1).value_counts()
renal_chronic=(datas["RENAL_CHRONIC"]==1).value_counts()
other_disease=(datas["OTHER_DISEASE"]==1).value_counts()




#geting true values of above condition
tru_diabetes=diabetes[True]
tru_hipertension=hipertension[True]
tru_pregnent=pregnent[True]
tru_copd=copd[True]
tru_asthma=asthma[True]
tru_inmsupr=inmsupr[True]
tru_pneumonia=pneumonia[True]
tru_cardiovascular=cardiovascular[True]
tru_obesity=obesity[True]
tru_tobacco=tobacco[True]
tru_renal_chronic=renal_chronic[True]
tru_other_disease=other_disease[True]


conditions=[tru_diabetes,tru_hipertension,tru_pregnent,tru_copd,tru_asthma,tru_inmsupr,\
            tru_pneumonia,tru_cardiovascular,tru_obesity,tru_tobacco,\
                tru_renal_chronic,tru_other_disease]
conditions_name=np.array(['diabetes','hipertension','pregnent','copd','asthma','inmsupr',\
                          'pneumonia','cardiovascular','obesity','tobacco','renal_chronic',\
                            'other_disease'])
plt.pie(conditions,labels=conditions_name,startangle=0,autopct='%1.1f%%',shadow=True)
plt.legend(conditions_name,loc="best")
plt.axis('equal')
plt.show()
print(conditions)
print(other_disease)
print(tru_other_disease)
print(tru_diabetes)
#conclustion:20.1%-hipertention,19.8%-obesity,17.3%-pneumonia,15.5%diabetes,10.4%-consume tobacco


#4 Can we see a trend over time in the number of cases and fatalities(death) due to COVID-19?
"""
death_data=datas["DATE_DIED"].value_counts()
print(death_data)
death_data_index=death_data.index
death_data_values=death_data.values
print(death_data_index)
print(death_data_values)
plt.barh(death_data_index,death_data_values)
plt.show()
"""

#5 What is the ratio of hospitalized patients to those who returned home?
#1-return home,2- hospitalisatin
patient_status=datas["PATIENT_TYPE"].value_counts()
print(patient_status)
returned_home=patient_status.values
hospitalised=patient_status.index
print(returned_home)
#print(hospitalised)
plt.bar(hospitalised,returned_home)
plt.xlabel("hospitalised/returned_home")
plt.ylabel("number of patients")
plt.show()
#conclustion:more number of patients are hospitalised.
#  it is of ratio 85000:3000 (8.5:3)aproximation(848544:200031)


#6 How many patients required intubation, and how does this correlate with the severity of
#the cases (based on the classification)?
intubation=(datas["INTUBED"]==1).value_counts()
print(intubation)
tru_intubation=intubation[True]
print("number of patients required intubation: ",tru_intubation)

severity=datas["CLASIFFICATION_FINAL"].value_counts()
print(severity)
correlate=tru_intubation & severity
correlate_index=correlate.index
correlate_values=correlate.values
print(correlate)
print(correlate_index)
print(correlate_values)
plt.bar(correlate_index,correlate_values,color="r")
plt.xlabel("status of patients")
plt.ylabel("number of required intibation")
plt.show()
#conclustion:1) 33656 patients required intubation
#            2)as we can seen  patients with higher diagnosed with 
#              covid requires intibation (here at level 3 need  intibation)


#7 What are the common comorbidities found in patients who died from COVID-19?




#******************Intermediate Level Questions:**********************
#1 Is there a correlation between age and the COVID-19 severity classification?
"""
print(age_counts)
print(severity)
age_covid=severity
print(age_covid)
#x=np.random.normal(0,20,300)
x=severity.values
plt.hist(x,range=10,color="r")
plt.xlabel("age range")
plt.ylabel("covid-19 severity")
plt.show()
#plt.plot(age_index,age_covid)
#plt.show()

"""

#2 How does the hospitalization rate vary with different pre-existing conditions?
#geting true values of above condition

#hospitalised with pre conditions
hos_diabetes=(patient_status[2]) & (tru_diabetes)
hos_hipertension=(patient_status[2]) & (tru_hipertension)
hos_pregnent=(patient_status[2]) & (tru_pregnent)
hos_copd=(patient_status[2]) & (tru_copd)
hos_asthma=(patient_status[2]) & (tru_asthma)
hos_inmsuprs=(patient_status[2]) & (tru_inmsupr)
hos_pneumonia=(patient_status[2]) & (tru_pneumonia)
hos_cardiovascular=(patient_status[2]) & (tru_cardiovascular)
hos_obesity=(patient_status[2]) & (tru_obesity)
hos_tobaccos=(patient_status[2]) & (tru_tobacco)
hos_renal_chronic=(patient_status[2]) & (tru_renal_chronic)
hos_other_disease=(patient_status[2]) & (tru_other_disease)


hos_conditions=[hos_diabetes,hos_hipertension,hos_pregnent,hos_copd,hos_asthma,\
                hos_inmsuprs,hos_pneumonia,hos_cardiovascular,hos_obesity,hos_tobaccos,\
                  hos_renal_chronic,hos_other_disease]
name_hos_conditions=["diabetes","hipertension","pregnent","copd","asthma","inmsuprs",\
                     "pneumonia","cardiovascular","obesity","tobaccos","renal_chronic",\
                      "other_disease"]
print(hos_diabetes)
print(hos_conditions)


plt.barh(name_hos_conditions,hos_conditions,color="gray")
plt.show()
plt.plot(name_hos_conditions,hos_conditions,marker="*",color="green",mec='r')
plt.show()
#conclustion:patients with pre-existing conditions hipertenstion,pnemonia,and obesity
#  requires hospitalization

#3 Is there a difference in the death rate between males and females?
#sex count 1-female,2-male
sex_counts=datas["SEX"].value_counts()
print(sex_counts)
#death count
death=(datas["DATE_DIED"]!="999-99-99").value_counts()
print(death)
female_death=(sex_counts[1]) & (death)
male_death=(sex_counts[2]) & (death)
print(female_death)
print(male_death)
sex_death=[male_death[True],female_death[True]]
name_sex=["male","female"]
print(sex_death)
print(name_sex)

plt.bar(name_sex,sex_death,color="orange")
plt.xlabel("sex")
plt.ylabel("number of deaths")
plt.show()

plt.plot(name_sex,sex_death,marker="o",color="r",mec='b')
plt.xlabel("sex")
plt.ylabel("number of deaths")
plt.show()
#conclustion:difference in the death rate between males and females is 1400
# death rate of females is higher than males
#it ranges :523600-525000

"""
#4 What is the relationship between obesity and the need for intensive care (ICU) or
#   ventilation support?
tru_obesity=obesity[True]
tru_intubation=intubation[True]
icu=(datas['ICU']==1).value_counts()
tru_icu=icu[True]
print(icu)
print(tru_icu)
obe_icu_or_venti=(tru_obesity & (tru_intubation | tru_icu))
print(obe_icu_or_venti)"""
#5 Does the type of medical unit (first, second, or third level) affect patient outcomes, such
#  as death or need for ICU?

#6 How does tobacco use impact the severity of COVID-19 and the outcome?
"""
print(tru_tobacco)
severity=datas["CLASIFFICATION_FINAL"].value_counts()

print(severity)
#outcome dead or recovered
death=(datas["DATE_DIED"]!="999-99-99").value_counts()
tru_death=death[True]
print(tru_death)


recovered=(datas["DATE_DIED"]=="999-99-99").value_counts()
tru_recovered=recovered[True]
print(tru_recovered)
"""
#7 Analyze the survival rate by comparing different classifications of COVID-19 with the
# presence of comorbidities.
death=(datas["DATE_DIED"]!='9999-99-99').value_counts()
print(death)
sur=datas["DATE_DIED"].value_counts()

survival=sur.iloc[0]
print(survival)

#classificatins 
clasi1=(datas["CLASIFFICATION_FINAL"]==1).value_counts()
clasi2=(datas["CLASIFFICATION_FINAL"]==2).value_counts()
clasi3=(datas["CLASIFFICATION_FINAL"]==3).value_counts()
tru_clasi1=clasi1[True]
tru_clasi2=clasi2[True]
tru_clasi3=clasi3[True]
print(clasi1,clasi2,clasi3)
print(tru_clasi1,tru_clasi2,tru_clasi3)

s1=tru_clasi1 & survival
s11=tru_clasi1 & death
s2=tru_clasi2 & survival
s21=tru_clasi2 & death
s3=tru_clasi3 & survival
s31=tru_clasi3 & death
print(s1)
print(s31)
print(s11)
print(s3)

tru_s1=s1
tru_s11=s11[True]
tru_s2=s2
tru_s21=s21[True]
tru_s3=s3
tru_s31=s31[True]
print(tru_s1)
print(tru_s31)
status_sd=[tru_s1,tru_s11,tru_s2,tru_s21,tru_s3,tru_s31]
status_name=["status1,survive","status1,death","status2,survive",\
             "status2,death","status3,survive","status3,death"]
my_color=['r','r','g','g','b','b']
print(status_sd)
print(status_name)
plt.bar(status_name,status_sd)
plt.xlabel("sttus/ condition of patients")
plt.ylabel("chances of survivalencce")
plt.show()
myexplod=[0,0,0,0,0.2,0]
plt.pie(status_sd,labels=status_name,startangle=1800,autopct='%1.1f%%',shadow=False,explode=myexplod)
plt.legend(conditions_name,loc="best")
plt.axis('equal')
plt.show()
#conclustion:status=3 class mean that the patient was diagnosed have nearly 80% 
#  chance of survival and 17.6% chance of not surviving 
