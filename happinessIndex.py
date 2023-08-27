import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import warnings 
warnings.filterwarnings("ignore")
df=pd.read_csv('/kaggle/input/happiness-index-2018-2019/report_2018-2019.csv')

#checking the country with highest happiness score
a=df.sort_values(by='Score',ascending=False)[['Country or region','Score','Year']]
a1=a.head(10)

a11=a1.loc[a1['Year']==2018]
index_to_drop=267
a11=a11.drop(index_to_drop)

a12=a1.loc[a1['Year']==2019]
index_to_drop=197
a12=a12.drop(index_to_drop)

trace1 = go.Bar(
    x=a11['Country or region'],
    y=a11['Score'],
    name='2018'
)

trace2 = go.Bar(
    x=a12['Country or region'],
    y=a12['Score'],
    name='2019'
)

# Create the layout for the double bar graph
layout = go.Layout(
    title='Comparison of countries with highest happiness score in each year',
    xaxis=dict(title='Country or region'),
    yaxis=dict(title='Happiness score'),
    barmode='group'  # This groups the bars side by side
)

# Combine the traces and layout into a figure
fig = go.Figure(data=[trace1, trace2], layout=layout)

# Show the double bar graph
fig.show()

b=df.sort_values(by='GDP per capita',ascending=False)[['Country or region','Score','Year','GDP per capita']]
b1=b.head(10)
b11=b1.loc[b1['Year']==2018]
b12=b1.loc[b1['Year']==2019]
index_to_drop=[144,124]
b12=b12.drop(index_to_drop)
trace1 = go.Bar(
    x=b11['Country or region'],
    y=b11['Score'],
    name='2018'
)

trace2 = go.Bar(
    x=b12['Country or region'],
    y=b12['GDP per capita'],
    name='2019'
)

# Create the layout for the double bar graph
layout = go.Layout(
    title='gdp per capita of each country in each year',
    xaxis=dict(title='Countries'),
    yaxis=dict(title='GDP per capita'),
    barmode='group'  # This groups the bars side by side
)

# Combine the traces and layout into a figure
fig = go.Figure(data=[trace1, trace2], layout=layout)

# Show the double bar graph
fig.show()
sns.scatterplot(x='GDP per capita', y='Score',data=df, color='blue', marker='o')sns.scatterplot(x='Social support', y='Score',data=df, color='blue', marker='o')
sns.scatterplot(x='Healthy life expectancy', y='Score',data=df, color='blue', marker='o')
sns.scatterplot(x='Freedom to make life choices', y='Score',data=df, color='blue', marker='o')
sns.scatterplot(x='Generosity', y='Score',data=df, color='blue', marker='o')
df.drop(['Overall rank','Country or region','Year','GDP per capita','Social support',"Freedom to make life choices","Generosity","Perceptions of corruption"],axis=1,inplace=True)
wcss=[]
for i in range(1,10):
    km=KMeans(n_clusters=i,init='k-means++',random_state=4)
    km=KMeans(n_clusters=i,init='k-means++',random_state=4)
    km.fit(df)
    wcss.append(km.inertia_)
plt.plot(range(1,10),wcss)
plt.xlabel("number of cluster")
plt.ylabel("WCSS")
plt.show()
km1=KMeans(n_clusters=3,init='k-means++',random_state=4)
y_kmeans=km1.fit_predict(df)
y_kmeans
df=np.array(df)
plt.scatter(df[y_kmeans==0,0], df[y_kmeans==0,1],s = 100,c = 'red', label = 'Cluster1')
plt.scatter(df[y_kmeans==1,0], df[y_kmeans==1,1],s = 100,c = 'blue', label = 'Cluster2')
plt.scatter(df[y_kmeans==2,0], df[y_kmeans==2,1],s = 100,c = 'green', label = 'Cluster3')
plt.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], s = 100, c = 'black', label = 'Centroid')
plt.title("Clusters")
plt.xlabel('Happiness Score')
plt.ylabel('Healthy life expectancy')
plt.legend()
plt.show()
df= pd.DataFrame(df,columns=('Score','Healthy life expectancy'))
y=np.array(y_kmeans)
df['cluster'] = y
a['Healthy life expectancy'].value_counts()
b=df.loc[df['cluster']==1][["Score",'Healthy life expectancy']]
c=df.loc[df['cluster']==2][["Score",'Healthy life expectancy']]
b['Healthy life expectancy'].value_counts()
c['Score'].value_counts()