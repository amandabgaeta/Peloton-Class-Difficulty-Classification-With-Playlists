# imports
import numpy as np
import pandas as pd
import operator

import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score, plot_confusion_matrix
import matplotlib.pyplot as plt


artist_df = pd.read_pickle("../../data/pickled_dfs/artist_df.pkl")
artist_id_class_count = pd.read_pickle("../../data/pickled_dfs/larger_spot_artist_id_class_count.pkl")  # has artist features
songs_df = pd.read_pickle("../../data/pickled_dfs/songs_df.pkl")
master_song_df = pd.read_pickle("../../data/pickled_dfs/master_song_df.pkl") # has song features

#  Data cleanse
def label_class_diff_cat(classDifficulty):
    '''
    Label Class Difficulty Beginner, Intermediate, or Advanced for each Peloton class based on classDifficulty rating
    
    Inputs:
    - Peloton classDifficulty value - float value between 0 and 10
    
    Outputs:
    - Class Difficulty label Beginner, Intermediate, or Advanced
    '''   
    if classDifficulty < 7.55:
        return 'Beginner'
    if classDifficulty >= 7.55 and classDifficulty < 8.37:
        return 'Intermediate'
    if classDifficulty >= 8.37:
        return 'Advanced'


# Model evaluation
def eval_model(estimator, X_train, X_test, y_train, y_test, target_names, average):
    '''
    Evaluation function to show accuracy, f1 score, mean 3-fold cross-validation 
    for both the train and test set, then shows confusion matrix for the test set
    
    Inputs:
    - estimator: model
    - X_train, y_train, X_test, y_test - train and test set results from train_test_split 
    - target_names - class labels to be used in classification report and confusion matrix
    - average - for setting f1 score details (micro, macro)
    
    Outputs:
    - Classification report
    - Training set Accuracy score
    - Training set F1 score with average (micro, macro) consideration
    - Test set Accuracy score
    - Test set F1 score with average (micro, macro) consideration
    - Test set F1 score Mean 3-Fold Cross Validation score
    - Confusion Matrix
    '''   
    # grab predictions
    train_preds = estimator.predict(X_train)
    test_preds = estimator.predict(X_test)
    
    # print report
    print(classification_report(y_test, test_preds, target_names=target_names))
    
    # print scores
    print("Train Scores")
    print("------------")
    print(f"Accuracy: {accuracy_score(y_train, train_preds)}")
    print(f"F1 Score: {f1_score(y_train, train_preds, average=average)}")
    print("----" * 5)
    print("Test Scores")
    print("-----------")
    print(f"Accuracy: {accuracy_score(y_test, test_preds)}")
    print(f"F1 Score: {f1_score(y_test, test_preds, average=average)}")
    print(f"F1 Score Mean Cross Val 3-Fold: {np.mean(cross_val_score(estimator,  X_train, y_train, cv=3, scoring=(f'f1_{average}')))}")
    
    # plot test confusion matrix
    plot_confusion_matrix(estimator, X_test, y_test, display_labels = target_names, values_format='')
    plt.show()


# Artist and song analysis
def diff_top_artists_and_songs(processed_data, orig_data):
    """
    Function takes in processed and original data, then merges this data with artist and song features
    to produce two DataFrames:
    1. Top artists
    2. Top songs 
    Both DataFrames include relative features, class counts, and class percentages (class_per) to 
    know of the number of classes where these artist or songs appear in the given data subset, and
    what percentage these counts makes up versus the full dataset of classes.
    -------
    Prerequisites:
     - Import pickled data sets:
        - artist_df - pd.read_pickle("../../data/pickled_dfs/artist_df.pkl")
        - artist_id_class_count (has artist features) - pd.read_pickle("../../data/pickled_dfs/larger_spot_artist_id_class_count.pkl")
        - songs_df - pd.read_pickle("../../data/pickled_dfs/songs_df.pkl")
        - master_song_df (has song features)- pd.read_pickle("../../data/pickled_dfs/master_song_df.pkl") 
    -------
    Input: 
    - processed_data - processed data used in winning model, now being used for analysis.
    This should be a subset of data and was built for the purpose to analyze different difficulties, 
    but can be used on any subset.
    - orig_data - original data set, pre-processing (pre scalers, OHE, etc)
    -------
    Output:
    - artists_with_feats - DataFrame of top artists with relative features
    - songs_with_feates - DataFrame of top songs with relative features
    """
    # artist_with_feats df creation
    # Merge processed and original data
    traceback = processed_data.merge(orig_data, how='left',left_index=True, right_index=True)
    
    # Create master subset
    orig_data_sub = traceback[['classId', 'classSongs', 'classArtists', 'popularity_song_y', 'explicit_y', 
                           'danceability_y', 'energy_y', 'key_y', 'loudness_y', 'mode_y', 'speechiness_y',
                           'acousticness_y', 'instrumentalness_y', 'liveness_y', 'valence_y', 'tempo_y', 
                           'time_signature_y', 'followers_y', 'popularity_artist_y', 'duration_mins_y']]
    
    
    # Merge with artist_df to get artists for original data
    orig_data_sub_with_artists = orig_data_sub.merge(artist_df, how='left', on='classId')
    
    # Create list of artists
    artist_list = list(orig_data_sub_with_artists.columns[21:])
    
    # Create artist dictionary using for loop to grab column name as the dict key and sum of each column as dict value
    artist_total_dict = {}

    for artist in artist_list:
        artist_total_dict[artist] = orig_data_sub_with_artists[artist].sum()

    sorted_artist_count_dict = dict(sorted(artist_total_dict.items(), key=operator.itemgetter(1),reverse=True))

    artist_count_df = pd.DataFrame.from_dict(sorted_artist_count_dict, orient='index').reset_index()
    artist_count_df.rename(columns={"index": "name", 0: "classCount"}, inplace = True)
    artist_count_df = artist_count_df[artist_count_df['classCount'] > 0]
    
    # Create new df of artists and counts with their artist features
    artists_with_feats = artist_count_df.merge(artist_id_class_count, how='inner', on='name')

    # Calculate % of classes this artist is used in that difficulty
    artists_with_feats.insert(4, 'class_per', round(artists_with_feats['classCount']/artists_with_feats['Class Count'], 2))
    
    # Filter down DataFrame to artists with classCount in 99th percentile
    artists_with_feats = artists_with_feats[artists_with_feats['classCount'] > (np.percentile(artists_with_feats['classCount'], 95))]
    
    # Sort by class_per to get to artists that have the highest percentage of their playlist
    # inclusion within this subset
    artists_with_feats = artists_with_feats.sort_values(by='class_per', ascending=False)
    
# ----------------------------------
    # songs df creation
    orig_data_sub_with_songs = orig_data_sub.merge(songs_df, how='left', on='classId')

    # Create list of song
    song_list = list(orig_data_sub_with_songs.columns[22:])

    # Create dictionary using for loop to grab column name as the dict key and sum of each column as dict value
    song_total_dict = {}

    for song in song_list:
        song_total_dict[song] = orig_data_sub_with_songs[song].sum()

    sorted_song_count_dict = dict(sorted(song_total_dict.items(), key=operator.itemgetter(1),reverse=True))

    song_count_df = pd.DataFrame.from_dict(sorted_song_count_dict, orient='index').reset_index()
    song_count_df.rename(columns={"index": "peloton_song_name", 0: "classCount"}, inplace = True)
    song_count_df = song_count_df[song_count_df['classCount'] > 0]

    # get what you need out of master song df to rematch up to song_count_df
    subset_master_songs = master_song_df[['peloton_song_name', 'song_id', 'Class Count', 'artist', 'artists', 'popularity', 'duration_ms', 'explicit', 'release_date', 
                                        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
                                        'instrumentalness', 'liveness', 'valence', 'tempo','time_signature']]
    
    # Create new df of songs and counts with their song features
    songs_with_feats = song_count_df.merge(subset_master_songs, how='left', on='peloton_song_name')

    # Calculate % of classes this song is used in that are in this subset
    songs_with_feats.insert(4, 'class_per', round(songs_with_feats['classCount']/songs_with_feats['Class Count'], 2))
    
    # Filter down DataFrame to songs with classCount in 99th percentile
    songs_with_feats = songs_with_feats[songs_with_feats['classCount'] > (np.percentile(songs_with_feats['classCount'], 95))]
    
    # Sort by class_per to get to songs that have the highest percentage of their playlist
    # inclusion within this subset
    songs_with_feats = songs_with_feats.sort_values(by='class_per', ascending=False)
    
    return artists_with_feats, songs_with_feats


# Plotting
def plot_scatter(data, x, y, hue, legend, artist_or_song, subset_title):
    plt.figure(figsize=(20,10))

    sns.scatterplot(data= data, x= x, y= y, hue= hue, legend= legend)


    plt.title((f'{artist_or_song} Class Count vs Class Percentage in {subset_title}'), fontsize=20, fontweight="bold")
    plt.xlabel((f'{artist_or_song} Class Count'), fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel((f'{artist_or_song} Class Percentage'), fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()