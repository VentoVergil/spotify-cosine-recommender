import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Analysis of spotify data from December 2021 to June 2022

pd.options.display.width = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 100

# Loading stream history
d1 = pd.read_json(r'Spotify\StreamingHistory0.json')
d2 = pd.read_json(r'Spotify\StreamingHistory1.json')

# Reading and loading liked songs in my library ('Your Library')
with open(r'Spotify\YourLibrary.json') as f:
    cc = json.load(f)
data1 = pd.DataFrame(cc['tracks'])

# Merging both stream histories into one dataframe
data2 = pd.concat([d1,d2])
assert len(d1) + len(d2) == len(data2),'Error'  # Sanity check, just to confirm that merge was successful

data1['uri'] = data1['uri'].str.split('spotify:track:').str[1]
data1['uid'] = data1['artist'] + ':' + data1['track']

# We are focusing on the streaming history of songs in our library
stream = data2[data2['trackName'].isin(data1['track'])]

# we can see that the liked songs counted for just a little above 50% of all user streams
print(f'{round((len(data2) - len(stream)) / len(data2) * 100)}%')

# Adding [track uri,album] column from the library dataframe to the created streamed dataframe
stream = stream.merge(data1,left_on = ['artistName','trackName'],right_on = ['artist','track'],how = 'right')

# Converting the stream time from milisecond to minutes
stream['minutes_played'] = round(stream['msPlayed'] / 60000,2)

# Splitting the endtime column into date and time columns
stream.insert(0,'date',value = pd.to_datetime(stream['endTime'].str.split().str[0]))
stream.insert(1,'time',value = stream['endTime'].str.split().str[1])

# creating day and month columns from the date columns
stream['day'] = stream['date'].dt.day_name()
stream['month'] = stream['date'].dt.month_name()

# Dropping the endtime and msplayed columns
stream.drop(['endTime','msPlayed','track','artist'],axis = 1,inplace = True)

# setting the date column as the index of the dataframe
stream.set_index('date',inplace = True)

# sorting the index in ascending order of dates
stream.sort_index(inplace = True)

# filtering out streams that were less than 30 seconds
stream.query('minutes_played > 0.3',inplace = True)

total_hours = []
month = []
months = stream.groupby('month',sort = False)

for group,data in months:
    month.append(group)
    total_hours.append(round(data['minutes_played'].sum() / 60))

# Visualizing total stream time by month
axis = np.arange(len(month))
plt.bar(month,total_hours,label='Total',width=0.6,color='g',edgecolor='black')
plt.xlabel('Month',fontweight='bold')
plt.ylabel('Total Hours',fontweight='bold')
plt.title('Total Listening Hours',fontsize=12,fontweight='bold')
plt.xticks(ticks = axis,labels = month,fontsize=9,rotation=30,fontweight='bold')
plt.tight_layout()
plt.savefig('Monthly Listening Hours.png')

# Reducing the length of some track and artist names to enable better plot
stream.replace(to_replace = ['Machine Gun Kelly',"Monica Lewinsky (feat. A Boogie Wit da Hoodie)",
                             'ay! (feat. Lil Wayne)'],value = ['MGK','Monica Lewinsky','ay!'],inplace = True)

# Finding top 10 songs with respect to total streaming hours
axis = np.arange(10)
table = stream.pivot_table(index = ['trackName'],values = 'minutes_played',aggfunc = 'sum')
table.sort_values(by = 'minutes_played',inplace = True,ascending = False)
table['minutes_played'] = table['minutes_played'] / 60
table = table['minutes_played'].round().astype(int)

sns.barplot(x = table.head(10).index,y = table.head(10).tolist(),edgecolor = 'black')
plt.xticks(ticks = axis,labels = table.head(10).index,fontsize = 8,rotation = 30,fontweight = 'bold')
plt.xlabel('Track',fontsize = 12,fontstyle = 'italic',fontweight = 'bold')
plt.ylabel('Total Hours',fontsize = 12,fontstyle = 'italic',fontweight = 'bold')
plt.title('Top Tracks by Total Stream Time (Dec 21 to June 22)',fontsize = 12)
plt.tight_layout(pad = 2.0)
plt.savefig('Top Songs by Stream Time.png')

# Finding top 10 artists with respect to total streaming hours
axis = np.arange(10)
table1 = stream.pivot_table(index = ['artistName'],values = 'minutes_played',aggfunc = 'sum')
table1.sort_values(by = 'minutes_played',inplace = True,ascending = False)
table1['minutes_played'] = table1['minutes_played']/60
table = table1['minutes_played'].round().astype(int)
sns.barplot(x=table.head(10).index,y=table.head(10).tolist(),edgecolor='black')

for i,v in enumerate(table.head(10).tolist()):
    plt.annotate(str(v),(i-0.195,v))

plt.xticks(ticks = axis,labels = table.head(10).index,fontsize=9,rotation=30)
plt.xlabel('Artists',fontsize=12,fontstyle='italic',fontweight = 'bold')
plt.ylabel('Total Hours',fontsize=12,fontstyle='italic',fontweight = 'bold')
plt.title('Top Artists by Total Stream Time (Dec 21 to June 22)',fontsize=12,fontweight = 'bold')
plt.tight_layout(pad = 2.0)
plt.savefig('Top Artists by Stream Time.png')

# Comparing Caraveo stream hours against total stream hours
axis = np.arange(len(month))
total_hours_cara = [0]
mont_cara = ['December']
caraveo = stream[stream['artistName'] == 'Ryan Caraveo']
caraveo_months = caraveo.groupby('month',sort = False)

for c_group,c_data in caraveo_months:
    mont_cara.append(c_group)
    total_hours_cara.append(round(c_data['minutes_played'].sum() / 60))

plt.bar(axis,total_hours,label='Total',width=0.4,color='g',edgecolor='black')
plt.bar(axis+0.4,total_hours_cara,label='Caraveo',width = 0.4,color='b')
plt.xticks(ticks = axis+0.4,labels = month,fontsize=9)
plt.legend()
plt.grid(visible = True,axis = 'y',alpha=0.5,ls='--',c='k',animated=True)
plt.xlabel('Months',fontsize=12,fontstyle='italic',fontweight = 'bold')
plt.ylabel('Total Hours',fontsize=12,fontstyle='italic',fontweight = 'bold')
plt.title('Total Stream Hours VS Total Caraveo Stream Hours',fontsize=12)
plt.savefig('Total Stream Hours VS Total Caraveo Stream Hours.png')


# We find that Ryan Caraveo is the users top artists.
# A further analysis will be performed on the dataframe
# We will get audio features of all tracks in library and classify them
