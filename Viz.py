import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure 'data.csv' is in the same directory as this script.
try:
    df = pd.read_csv('data.csv', header=None)
except FileNotFoundError:
    print("Error: The file 'data.csv' was not found. Please ensure the file is in the same directory as your script.")
    exit()

# Based on the file snippet, the first column is an ID, the second is the diagnosis,
# and the remaining columns are the features.
df.drop(columns=0, inplace=True)
df.rename(columns={1: 'Diagnosis'}, inplace=True)

# The remaining columns are the 32 features.
features = df.columns[1:]

# Set up the plotting environment
sns.set_style("whitegrid")

# -----------------
# 1. Histograms
# -----------------
print("Generating histograms...")

# Create a figure with a grid of subplots for the 32 features
fig_hist, axes_hist = plt.subplots(nrows=8, ncols=4, figsize=(30, 60))
axes_hist = axes_hist.flatten()

# Loop through each feature and create a histogram
for i, col in enumerate(features):
    sns.histplot(data=df, x=col, hue='Diagnosis', multiple='stack',
                 bins=30, ax=axes_hist[i], kde=True)
    axes_hist[i].set_title(f'Feature {i+1} Distribution', fontsize=12)
    axes_hist[i].set_xlabel('')
    axes_hist[i].legend(title='Diagnosis', loc='upper right', labels=['Malignant', 'Benign'])

# Hide any extra subplots
for i in range(len(features), len(axes_hist)):
    fig_hist.delaxes(axes_hist[i])

# Adjust layout and save the plot as a PNG file
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()
print("Feature distribution plot saved as 'feature_distributions.png'.")

# -----------------
# 2. Box plots
# -----------------
print("Generating box plots...")

# Create a new figure and axes for the box plots
fig_box, axes_box = plt.subplots(nrows=8, ncols=4, figsize=(20, 40))
axes_box = axes_box.flatten()

# Loop through each feature and create a box plot
for i, col in enumerate(features):
    sns.boxplot(data=df, x='Diagnosis', y=col, ax=axes_box[i])
    axes_box[i].set_title(f'Feature {i+1} Box Plot', fontsize=12)
    axes_box[i].set_xlabel('')

# Hide any extra subplots
for i in range(len(features), len(axes_box)):
    fig_box.delaxes(axes_box[i])

# Adjust layout and save the plot as a PNG file
plt.tight_layout()
plt.savefig('feature_boxplots.png')
plt.show()
print("Feature boxplot saved as 'feature_boxplots.png'.")

# -----------------
# 3. Correlation Matrix
# -----------------
print("Generating correlation matrix heatmap...")

# Before calculating correlation, we must encode the diagnosis column to numerical values
df_corr = df.copy()
df_corr['Diagnosis'] = df_corr['Diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Calculate the correlation matrix
correlation_matrix = df_corr.corr()

# Create a new figure for the heatmap
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of All Features', fontsize=20)

# Save the plot
plt.savefig('correlation_matrix.png')
plt.show()
print("Correlation matrix heatmap saved as 'correlation_matrix.png'.")