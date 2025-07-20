def plot_labels_dist(train_data):
    # Clip and plot the label distribution
    plt.hist(train_data['label'].clip(lower=-4, upper=4), bins=50)
    plt.title('Clipped Distribution of Label')
    plt.xlabel('Label Value')
    plt.ylabel('Frequency')
    plt.show()


def create_more_features(train_data):
    train_data.index.name = 'timestamp'
    train_data = train_data.reset_index()  # Ensure timestamp is a column
    train_data['lag_1'] = train_data['label'].shift(1)  # Previous label value
    train_data['lag_2'] = train_data['label'].shift(2)  # Previous previous label value
    print(train_data[['timestamp', 'label', 'lag_1', 'lag_2']].head())
    return train_data

