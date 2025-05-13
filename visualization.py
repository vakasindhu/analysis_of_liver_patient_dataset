# Visualizations
# 1 
plt.scatter(df.iloc[:, 8].values, df.iloc[:, 9].values, color = ['blue'])

# 2
plt.hist(df.iloc[:, [3]])
plt.show()

# 3
plt.figure(figsize=(33, 33))
sns.heatmap(copyDF, annot=True, fmt=".3f", linewidths=.5, square = True)

# 4
copyDF

# 5
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
age = copyDF['Age']
result = copyDF['Result']
ax.bar(age,result)
plt.show()

# 6
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
gender = copyDF['Gender']
result = copyDF['Result']
ax.bar(gender,result)
plt.show()

# 7
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
gender = copyDF['Gender']
result = copyDF['Result']
ax.bar(gender,result) plt.show()
