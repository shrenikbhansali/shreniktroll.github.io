Predictive Modeling for Spotify Popularity - Midterm Report
Google Colab: https://colab.research.google.com/drive/11mNFnZme0wsW4htZ1aJgK0gFAsJUZ-X8

Project Proposal Video: https://youtu.be/0RlFCZbYZgU

Timeline + Contribution Table: https://docs.google.com/spreadsheets/d/1P0Y_QHY5c5-IPdxbHxNGuAQMk4SuVZeS0ODhYT9_oM0/edit?usp=sharing


Data Collection
Data set: https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019
Top Hits Spotify from 2000-2019 is a dataset found on Kaggle’s website with a high usability score with 18 features and almost 2,000 unique values. This dataset contains various audio statistics that may contain patterns regarding what qualities of music make a song popular throughout the decade. Deciphering what characteristics make songs become a popular and create an enjoyable listening experience to the masses is important for many parties including artists, record label companies and stations who host the music for their audiences.
Below is a list of the data’s features downloaded from songs_normalize.csv:
- Artist: Name of the artist
- Song: Name of the track
- Duration_ms: Duration of the track in milliseconds
- Explicit: The lyrics or content of a song or a music video contain one or more of the criteria which could be considered offensive or unsuitable for children
- Year: Release year of the track
- Popularity: The value determines the popularity of the song-- a higher value indicates  a more popular song
- Danceability: Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable
- Energy: measure from 0.0 to 1.0 that represents a perceptual measure of intensity and activity
- Key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
- Loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
- Mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
- Speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- Instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but Confidence is higher as the value approaches 1.0.
- Acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- Liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.
- Valence: A measure from 0.0 to 1.0 describing the musical mood conveyed. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- Tempo: The average estimated tempo/pace of a track in beats per minute (BPM).
- Genre: Categorization of the music into a related genre.


Data Exploration
In order to gain a better understanding of the data, we wanted to explore the data from two perspectives-- descriptively and statistically. When descriptively viewing the data, we wanted to explore who was popular, what genres were popular, and who was producing the most popular music. The results of those questions are shown here:


IMAGESIMGAGESIMAGES

These findings help us understand that a popular artist is not the same as an artist who makes popular music. While Eminem might have more total popularity, he created less popular music since he had fewer top songs. Each of his songs were individually higher in popularity than Drake when summed up, but Drake had a greater total number of popular songs. The genre results also aided us in determining to not pursuing one-hot encoding and treating intersections of genres as their own unique genres, since all of the top 5 genres contained pop. For the sake of modeling, we thought it would be better to continue to let them be independent, in fear of the pop genre being the main determinant of popularity. In reality, the majority of songs in the data are labeled with pop in combination with another genre.

Statistically, we wanted to explore the data and their distributions to notice any patterns or anomalies that would let us better analyze it and select methods. The statistical exploration is below:

BIGIMAGE
Although not much can be inferred from viewing a single distribution, the juxtaposition of the figures demonstrates that much of the data follows gaussian distributions. This was taken into consideration when pursuing dimensionality reductions and when selecting methods to pursue (methods like LDA may perform better). It also showed a lack of outliers, and it's worth noting that not all of the data had the similar distributions (meaning methods like LDA could also not perform well in certain cases).





Data Cleaning and Preprocessing
Of the Spotify metrics listed above, we tried to keep as many as possible when manually selecting features. Without analyzing the variance of them, the dataset seemingly contained relevant features that would be useful for our popularity analysis. Many of the Spotify metrics could not have been manually identified as important or unimportant. However, song name and song release year did come into question. We felt that the name of the song was less important since very rarely does a song become popular due to its name. Usually, songs become popular because of the musical quality, which is measured in various feature types in the Spotify metrics. Additionally, some songs become popular solely because of the artist who produced it or the genre that it is categorized under. For those reasons, we chose to include artists and genres as features. We also felt song year should not be disregarded as many songs that were released in certain eras could be more popular than others. Thus, the only feature that was manually dropped was the song name.

During the data cleaning phase, we enountered string data types that weren’t the most efficient or useful. Artist names, genres, and explicitness were all examples of data in string format. In order to resolve this, we determined fitting encoding schemes for the data. The binary nature of explicit and not explicit allowed us to simply toggle between 1 and 0 for explicit and non-explicit, respectively. For genre and artists, we chose simple integer encoding instead of one-hot encoding since the data was not presented in a way where songs could have artists at one time. While they could have multiple genres at one time, we felt it would be better to treat intersections of genres as their own respective genres.  Doing this, we preserved the encodings to create dictionaries that reverse the encoding, in case we want to reverse the encoding when looking at certain results. This means while doing our exploratory analysis of the data, we were able to recover artist names and genres. 

In terms of data preprocessing, we were still left with 16 features, which would present obstacles in data visualizations and potentially in computation. In order to decide the optimal amount of features to use, we decided to analyze the variance of the data across the number of components, with the resulting graph below:

IMAGE

The graph illustrates that the amount of variance is highly correlated with the amount of components, not giving us a clear point to cut them off. Because of this, at this stage in the project, we decided to create 3 versions of the data. We currently have one version with all 16 features, one with 10 and one with 3. We chose these because 16 features would theoretically give us the highest accuracy since it covers the most variance, but at a risk of overfitting. 10 features covers roughly 80% of the variance of the data, while reducing the amount of features for both computational efficiency and reduce the risk of overfitting. Lastly, we selected 3 features solely for visualization purposes. Below is the relative amount of variance in each of the sets of data:

IMAGE
In order to perform our dimensionality reduction to acquire our data with 3 and 10 features, we started by normalizing the data so it could better be analyzed. We performed PCA to capture 80% of our data variance, which resulted in retrieving 10 features. 80 percent was an arbitrary number that was picked to represent a structuring of data that captures a sizable amount of variance but not all of it. We chose PCA by comparing its performance against LDA and Lasso, which when run with cross validation testing, gave us the best results relative to the other two techniques.

For the data with 3 features however, it was a different story. LDA yielded us the best version of the data, which may be because some of the features generally followed a gaussian distribution as mentioned in the data exploration section. The corresponding scatterplot for LDA is below:

IMAGE

After testing more methods on each of these sets of data, we will analyze the preliminary results in order to decide which set of data to use consistently throughout. The methods discussed below will be run on our PCA resultant data (10 features), our LDA resultant data (3 features), and our non-reduced data (16 features).



KMEANS stuff
