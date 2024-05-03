import csv
import boto3
import logging
import os
from datetime import datetime
from io import StringIO
import pandas as pd
import numpy as np

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Retrieve the environment variables
malicious_bucket_name = os.environ['MALICIOUS_BUCKET_NAME']
processed_bucket_name = os.environ['PROCESSED_BUCKET_NAME']
results_table_name = os.environ['RESULTS_TABLE_NAME']
user_table_name = os.environ['USER_TABLE_NAME']
analysis_table_name = os.environ['ANALYSIS_TABLE_NAME']

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#CONSTANTS
USERS_TO_PROCESS = {'OFS0030', 'RCG0584', 'BDP0096', 'HBP1076'}

def lambda_handler(event, context):
    """
    Main lambda function.

    Args:
        event: Event dictionary.
        context: Lambda context.

    Returns:
        Response object.
    """
    try:
        # Set up dynamodb table objects
        results_table = dynamodb.Table(os.environ['RESULTS_TABLE_NAME'])
        user_table = dynamodb.Table(os.environ['USER_TABLE_NAME'])
        analysis_table = dynamodb.Table(os.environ['ANALYSIS_TABLE_NAME'])

        # Fetch user IDs from the user table
        user_ids = get_all_users(user_table)

        # Load malicious email IDs (malicious IDs and their associated time periods)
        malicious_ids = load_malicious_periods(malicious_bucket_name, 'malicious_ids.csv')
        
        # Update malicious user periods
        update_users_malicious_periods(user_table, malicious_ids)

        # Iterate through the user ids
        for user_id in user_ids:
            # Skip users not wanted
            if user_id not in USERS_TO_PROCESS:
                logger.info(f"Skipping user {user_id} as not in USERS_TO_PROCESS.")
                continue
            try:
                # Obtain user data from results table and convert to dataframe for pandas processing.
                user_data = get_user_data(user_id, results_table)
                user_df = convert_to_df(user_data)
                print("Obtained User DF")
                


                # Calculate baseline statistics for user, by removing malicious data.
                baseline_df = exclude_malicious_data(user_df, malicious_ids)
                print("Obtained Baseline DF")
                baseline_stats = calculate_baseline_stats(baseline_df)
                print("Obtained Baseline Stats")

                # Insert the baseline statistics into the analysis table.
                update_baseline_records(user_id, baseline_stats, analysis_table)

                # Create the daily and weekly stats for the user and normalize the data.
                daily_df = calculate_daily(user_df)
                weekly_df = calculate_weekly(user_df)

                # Calculate the moving averages for the user.
                daily_df = calculate_daily_rolling_averages(daily_df)
                weekly_df = calculate_weekly_rolling_averages(weekly_df)

                # Calculate the z-scores for the user, using the baseline stats.
                daily_stats = calculate_daily_z_scores(daily_df, baseline_stats)
                weekly_stats = calculate_weekly_z_scores(weekly_df, baseline_stats)

                # Update the analysis table with the daily and weekly stats for the user.
                update_daily_analysis_table(daily_stats, analysis_table)
                update_weekly_analysis_table(weekly_stats, analysis_table)
            
            # Handle errors for user
            except Exception as e:
                logger.error(f'Error carrying out anomaly analysis of {user_id}: {str(e)}')

        # Log success
        logger.info('Anomaly analysis completed successfully')
        return {
            'statusCode': 200,
            'body': 'Anomaly analysis completed successfully'
        }
    # Handle general processing errors
    except Exception as e:
        logger.error(f'Error carrying out anomaly analysis of: {str(e)}')
        return {
            'statusCode': 500,
            'body': 'Error carrying out anomaly analysis'
        }

def get_all_users(user_table):
    """
    Retrieves all user IDs from users table.

    :param user_table: The name of user table.
    :return: List of user IDs.
    """

    # Empty list for user ids
    user_ids = []
    
    # Obtain all user ids. Whether to use scan or query doesn't make a difference, as there is only one attribute in the table. The table isn't strictly necessary, but it was to ease implementation. Probably more resource expensive.
    response = user_table.scan(AttributesToGet=['userid'])
    
    # Collect all user IDs from the scan response
    for item in response['Items']:
        user_ids.append(item['userid'])
    
    # Handle potential pagination if the scan result is truncated
    while 'LastEvaluatedKey' in response:
        response = user_table.scan(AttributesToGet=['userid'], ExclusiveStartKey=response['LastEvaluatedKey'])
        for item in response['Items']:
            user_ids.append(item['userid'])

    # Return the user ids
    return user_ids

def get_user_data(user_id, results_table):
    """
    Fetches all data for a specific user from the results table.

    :param user_id: User ID.
    :param results_table: Results table.
    :return: A list of dictionaries for the specified user.
    """
    try:
        # Query the data for the user from the results table.
        response = results_table.query(
            KeyConditionExpression='userid = :user_id',
            ExpressionAttributeValues={':user_id': user_id}
        )
    except Exception as e:
        print(f"Error fetching data for user {user_id}: {str(e)}")
        return []

    # Accumulate items from the query result.
    items = response['Items']

    # Handle paginated results: if LastEvaluatedKey exists, continue querying.
    while 'LastEvaluatedKey' in response:
        try:
            response = results_table.query(
                KeyConditionExpression='userid = :user_id',
                ExpressionAttributeValues={':user_id': user_id},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            # Extend the item list with the newly fetched items.
            items.extend(response['Items'])
        except Exception as e:
            # Log pagination error and break from the loop; return what has been accumulated so far.
            logger.error(f"Error fetching subsequent data for user {user_id} with pagination: {e}")
            break

    # Return the items
    return items

def normalize_sentiment(sentiment, confidence):
    """
    Normalizes the sentiment value based on its category and associated confidence.

    Multiplies a predefined scale factor for the sentiment category
    by the confidence level to obtain a weighted sentiment score. Score reflects
    the type and strength of the sentiment.
    
    :param sentiment (str): The sentiment category (POSITIVE, NEGATIVE, NEUTRAL, MIXED).
    :param confidence (float): The confidence value of the sentiment analysis from Comprehend, between 0 and 1.
    :return: A normalized value for the sentiment, or 0 if an invalid input is passed.
    """

    # # Validate sentiment input.
    # if not isinstance(sentiment, str) or sentiment.upper() not in {'POSITIVE', 'NEGATIVE', 'NEUTRAL', 'MIXED'}:
    #     logger.error(f"Invalid sentiment value: {sentiment}")
    #     return 0
    
    # # Validate confidence input.
    # if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
    #     logger.error(f"Invalid confidence value: {confidence}")
    #     return 0

    # Define sentiment scale and compute the normalized sentiment score.
    sentiment_scale = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0, 'MIXED': 0}
    try:
        # Try and return the normalized sentiment score
        return sentiment_scale[sentiment.upper()] * confidence
    except Exception as e:
        # Log any errors that occurs during sentiment normalization.
        logger.error(f"Unexpected error during sentiment normalization: {e}")
        return 0 # Return zero if errors occur

def convert_to_df(data):
    """
    Converts the user data into a DataFrame.
    Normalizes sentiment data.
    Expands the pronoun data by categorising them into first person singular, first person plural, and second person. So that appropriate trends can be calculated.
    Calculates the total frequency of the cognitive terms.

    :param data: User data pulled from the results table.
    :return: DataFrame with individual columns for all attributes.
    """

    # Create the initial dataframe from the data.
    df = pd.DataFrame(data)

    # Normalizing sentiment values
    df['Sentiment'] = df.apply(lambda row: normalize_sentiment(row.get('sentiment'), row.get('confidence', 0)), axis=1)

    # Initialize columns for pronoun categories even if they are absent in the data
    pronoun_categories = {
        'FirstPersonSingular': ['pronoun_I', 'pronoun_me', 'pronoun_my', 'pronoun_mine', 'pronoun_myself'],
        'FirstPersonPlural': ['pronoun_we', 'pronoun_us', 'pronoun_our', 'pronoun_ours', 'pronoun_ourselves'],
        'SecondPerson': ['pronoun_you', 'pronoun_your', 'pronoun_yours', 'pronoun_yourself', 'pronoun_yourselves']
    }

    # If pronouns are present, expand them into columns
    if 'pronouns' in df and not df['pronouns'].isnull().all():
        # Expand nested dictionary of pronouns into separate columns. pd.series is basically just a dictionary.
        pronoun_df = df['pronouns'].apply(pd.Series)
        # Add the prefix pronoun_ to identify the columns for the pronouns
        pronoun_columns = pronoun_df.add_prefix('pronoun_')
        # Add the pronoun columns to the original dataframe.
        df = pd.concat([df, pronoun_columns], axis=1)
    else:
        # If no pronouns are present, initialize columns with zeros
        for category, pronouns in pronoun_categories.items():
            for pronoun in pronouns:
                df[pronoun] = 0

    # Aggregate pronouns into categories
    for category, pronouns in pronoun_categories.items():
        # Create new dataframe column for pronoun category by summing the frequencies of the pronoun type
        # List comprehension iterates over pronouns list and includes the pronoun if it also is a column in df.
        # Axis=1 sums along the horizontal axis(across columns)
        df[category] = df[[pronoun for pronoun in pronouns if pronoun in df.columns]].sum(axis=1)

    # Cognitive terms processing
    # If cognitive terms are in df and not null.
    if 'cognitiveterms' in df and not df['cognitiveterms'].isnull().all():
        # Expand nested dictionary of cognitiveterms into separate columns. pd.series is basically just a dictionary.
        cognitive_terms_df = df['cognitiveterms'].apply(pd.Series)
        # Add cognitive terms column to user dataframe.
        df = pd.concat([df, cognitive_terms_df], axis=1)
        # Sum the cognitive term frequencies
        df['CognitiveTerms'] = cognitive_terms_df.sum(axis=1)
    else:
        # Else if there are no cognitive terms, set to zero
        df['CognitiveTerms'] = 0

    # Ensuring numeric columns are formatted correctly, convert any NaN values to zeros.
    numeric_columns = ['Sentiment', 'FirstPersonSingular', 'FirstPersonPlural', 'SecondPerson', 'CognitiveTerms']
    df[numeric_columns] = df[numeric_columns].fillna(0).astype(float)

    return df


def load_malicious_periods(bucket_name, file_key):
    """
    Loads the malicious email IDs from the specified file in the specified S3 bucket.

    :param bucket_name: Malicious IDs S3 bucket name.
    :param file_key: Malicious IDs file key.
    :return: A set containing the malicious periods.
    """
    try:
        # Retrieve the malicious IDs file from the bucket and load into pandas dataframe.
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        periods_df = pd.read_csv(obj['Body'], parse_dates=['start', 'end'])
    except Exception as e:
        logger.error(f"Error loading or processing file from S3: {e}")
        return {} # Return empty dictionary on error
    
    # Grouping by user using groupby(), all rows for a single user will be aggregated.
    # apply() then applies the lambda that takes in the user groups and pairs the start and end dates together.
    # This results in a dictionary of user IDs and associated start and end values.
    malicious_periods = periods_df.groupby('user').apply(
        lambda x: list(zip(x.start, x.end))).to_dict()
    
    # Return the dictionary of the malicious periods
    return malicious_periods


def exclude_malicious_data(user_df, malicious_periods):
    """
    Excludes malicious data that is within the malicious time periods for each user so that baseline can be calculated.

    :param user_df: DataFrame containing all user email records.
    :param malicious_ids: A set containing the identifiers of malicious emails.
    :return: DataFrame excluding the identified malicious records.
    """
    # Ensure 'date' in user_df is in datetime format for comparison
    user_df['date'] = pd.to_datetime(user_df['date'])

    # Convert start and end times in malicious_periods to datetime format
    for user, periods in malicious_periods.items():
        try:
            # Tuple of malicious periods is iterated over using the list comprehension and then the pandas to_datetime() function is applied to ensure the start and end dates are in the correct formats.
            # This is done to the column of the user in the dataframe.
            malicious_periods[user] = [(pd.to_datetime(start), 
                                        pd.to_datetime(end)) 
                                    for start, end in periods]
        except ValueError as e:
            print(f"Error converting date for user {user}: {e}")

    # Function to check if the record is malicious
    def is_malicious(row):
        """
        Determines if an email record falls within any malicious period for its user.

        :param row: A DataFrame row for the user email.
        :return: True for malicious, False for benign.
        """
        # Extract user ID and date
        user, timestamp = row['userid'], row['date']

        # Loop over the dates for the user, if user doesn't exist, return empty list.
        for start, end in malicious_periods.get(user, []):
            # Ensure start and end are datetimes
            if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                # If the record date lies within the malicious period, then return True for it being malicious.
                if start <= timestamp <= end:
                    return True
        # If not malicious, return False
        return False

    # Apply the function to determine if each record is malicious
    # Add new column to identify if record is malicious, apply is_malicious to each row in the user dataframe
    user_df['is_malicious'] = user_df.apply(is_malicious, axis=1)

    # Create new baseline dataframe for the baseline statistics.
    # Using boolean indexing to only select where is_malicious is False by using ~, ~ is a NOT operation.
    # Resulting baseline_df only contains non malicious records and drops the is_malicious column.
    baseline_df = user_df[~user_df['is_malicious']].drop(columns=['is_malicious'])

    logger.info("Malicious data identification complete.")

    return baseline_df
    
def update_users_malicious_periods(table, malicious_ids):
    """
    Updates user records in the DynamoDB table with malicious periods.

    :param user_table_name: Name of the DynamoDB user table.
    :param malicious_ids: Dictionary mapping user IDs to their malicious periods.
    """
        
    for user_id, periods in malicious_ids.items():
        if user_id in USERS_TO_PROCESS:
            try:
                for period in periods:
                    start_date, end_date = period  # Unpacking the tuple
                    # Update user record with the malicious periods
                    response = table.update_item(
                        Key={'userid': user_id},
                        UpdateExpression='SET malicious_periods = list_append(if_not_exists(malicious_periods, :empty_list), :new_period)',
                        ExpressionAttributeValues={
                            ':new_period': [{'start': start_date, 'end': end_date}],
                            ':empty_list': [],
                        },
                        ReturnValues='UPDATED_NEW'
                    )
            except Exception as e:
                logger.error(f"Error updating malicious period for user {user_id}: {e}")

def calculate_baseline_stats(baseline_df):
    """
    Calculates baseline metrics (mean and standard deviation) for each feature.

    :param baseline_df: DataFrame containing only benign data.
    :return: A dictionary containing baseline stats (mean and standard deviation).
    """

    baseline_stats = {}
    # Define the linguistic features we want.
    features = ['Sentiment', 'FirstPersonSingular', 'FirstPersonPlural', 'SecondPerson', 'CognitiveTerms']

    # Iterate over the features.
    for feature in features:
        # Calculate mean; Nan values are ignored.
        mean_val = baseline_df[feature].mean()
        
        # Calculate standard deviation for feature
        std_val = baseline_df[feature].std()

        # Check for NaN values using np.isnan() and set them to zero if found.
        if np.isnan(std_val):
            logger.warning(f"NaN standard deviation for {feature} is replaced with 0.")
            std_val = 0

        # Set the mean and std for each feature in the baseline_stats dictionary.
        baseline_stats[f'{feature}_mean'] = mean_val
        baseline_stats[f'{feature}_std'] = std_val

    return baseline_stats

def update_baseline_records(user_id, baseline_stats, table):
    """
    Stores the provided baseline statistics in the analysis table.

    :param baseline_stats: Dictionary containing the baseline stats.
    :param analysis_table_name: Analysis table name.
    """

    # Prepare the baseline statistics item for DynamoDB
    baseline_stats_item = {
        'userid': user_id,
        'date': 'baseline',
    }

    # Iterate over the baseline stats to add them to the item, ensuring all values are strings for compatibility with DynamoDB
    for stat_key, stat_value in baseline_stats.items():
        baseline_stats_item[stat_key] = str(stat_value)

    # Store the baseline statistics item in analysis table
    try:
        table.put_item(Item=baseline_stats_item)
        print(f"Baseline statistics successfully stored in table {analysis_table_name}.")
    except Exception as e:
        print(f"Error storing baseline statistics: {str(e)}")


def calculate_daily(df):
    """
    Calculates daily statistics from the email data and normalizes frequencies by email count.

    :param df: A DataFrame containing the user's email data.
    :return: A DataFrame with normalized daily statistics.
    """

    # Return an empty dataframe if no user data supplied.
    if df.empty:
        return pd.DataFrame()

    # Ensure the 'date' column is in the appropriate datetime format
    df['date'] = pd.to_datetime(df['date'])
    # Extract just the date component from the datetime
    df['just_date'] = df['date'].dt.date

    # Count emails per user per day by splitting into groups where both userid and date are the same, e.g. sent by user on that day.
    # Size() counts the number of rows in each group, e.g. number of emails that day
    # reset_index() is creating a column called email_count with the email count for each day for a certain user.
    email_counts = df.groupby(['userid', 'just_date']).size().reset_index(name='email_count')

    # Aggregate and normalize frequencies
    # Grouping for user per day again.
    # Apply aggregation functions for each column, e.g. we calculate the mean sentiment, and total pronoun types and cognitiveterms for that user's day.
    # Resets index to make userid and just_date into proper columns.
    daily_stats = df.groupby(['userid', 'just_date']).agg({
        'Sentiment': 'mean',
        'FirstPersonSingular': 'sum',
        'FirstPersonPlural': 'sum',
        'SecondPerson': 'sum',
        'CognitiveTerms': 'sum'
    }).reset_index()

    # Merge with the email counts so daily_stats has both email counts and daily_stats for each combination of userid and date. E.g. stats and email count for each day per user.
    daily_stats = daily_stats.merge(email_counts, on=['userid', 'just_date'])

    # Normalize pronouns and cognitive terms frequencies by dividing per email count per day.
    daily_stats['FirstPersonSingularDailyMean'] = daily_stats['FirstPersonSingular'] / daily_stats['email_count']
    daily_stats['FirstPersonPluralDailyMean'] = daily_stats['FirstPersonPlural'] / daily_stats['email_count']
    daily_stats['SecondPersonDailyMean'] = daily_stats['SecondPerson'] / daily_stats['email_count']
    daily_stats['CognitiveTermsDailyMean'] = daily_stats['CognitiveTerms'] / daily_stats['email_count']

    # Now drop the redundant original columns, in place, without returning a new dataframe. Axis=1 indicates the columns should be dropped not the rows.
    daily_stats.drop(['FirstPersonSingular', 'FirstPersonPlural', 'SecondPerson', 'CognitiveTerms', 'email_count'], axis=1, inplace=True)

    # Renaming 'Sentiment' column to clarify it represents a daily mean.
    daily_stats.rename(columns={'Sentiment': 'SentimentDailyMean'}, inplace=True)

    return daily_stats

def calculate_weekly(df):
    """
    Calculates weekly statistics from the user's email data and normalizes frequencies.

    :param df: A DataFrame containing the user's data.
    :return: A DataFrame with normalized weekly statistics.
    """

    # Return an empty dataframe if no user data supplied.
    if df.empty:
        return pd.DataFrame()

    # Ensure the 'date' column is in the appropriate datetime format
    df['date'] = pd.to_datetime(df['date'])
    # Extract the week number and year from the datetime
    df['year_week'] = df['date'].dt.strftime('%Y-W%V')

    # Count emails per user per week by splitting into groups where both userid and date are the same, e.g. sent by user that week.
    # Size() counts the number of rows in each group, e.g. number of emails that day
    # reset_index() is creating a column called email_count with the email count for each week for a certain user.
    email_counts_weekly = df.groupby(['userid', 'year_week']).size().reset_index(name='email_count_weekly')

    # Aggregate and normalize frequencies
    # Grouping for user per week again.
    # Apply aggregation functions for each column, e.g. we calculate the mean sentiment, and total pronoun types and cognitiveterms for that user's week.
    # Resets index to make userid and year_week into proper columns.
    weekly_stats = df.groupby(['userid', 'year_week']).agg({
        'Sentiment': 'mean',
        'FirstPersonSingular': 'sum',
        'FirstPersonPlural': 'sum',
        'SecondPerson': 'sum',
        'CognitiveTerms': 'sum'
    }).reset_index()

    # Merge with the email counts so weekly_stats has both email counts and weekly_stats for each combination of userid and year_week. E.g. stats and email count for each week per user.
    weekly_stats = weekly_stats.merge(email_counts_weekly, on=['userid', 'year_week'])

    # Normalize pronouns and cognitive terms frequencies by dividing per email count per week.
    weekly_stats['FirstPersonSingularWeeklyMean'] = weekly_stats['FirstPersonSingular'] / weekly_stats['email_count_weekly']
    weekly_stats['FirstPersonPluralWeeklyMean'] = weekly_stats['FirstPersonPlural'] / weekly_stats['email_count_weekly']
    weekly_stats['SecondPersonWeeklyMean'] = weekly_stats['SecondPerson'] / weekly_stats['email_count_weekly']
    weekly_stats['CognitiveTermsWeeklyMean'] = weekly_stats['CognitiveTerms'] / weekly_stats['email_count_weekly']

    # Now drop the redundant original columns, in place, without returning a new dataframe. Axis=1 indicates the columns should be dropped not the rows.
    weekly_stats.drop(['FirstPersonSingular', 'FirstPersonPlural', 'SecondPerson', 'CognitiveTerms', 'email_count_weekly'], axis=1, inplace=True)

    # Rename sentiment column for clarity
    weekly_stats.rename(columns={'Sentiment': 'SentimentWeeklyMean'}, inplace=True)

    return weekly_stats

def calculate_daily_rolling_averages(daily_stats, window_size=7):
    """
    Calculates rolling averages for daily statistics.

    :param daily_stats: DataFrame containing daily statistics.
    :param window_size: The size of the moving window.
    :return: DataFrame with rolling averages appended.
    """

    # Define the metrics we want
    metrics = [
        'SentimentDailyMean',
        'FirstPersonSingularDailyMean',
        'FirstPersonPluralDailyMean',
        'SecondPersonDailyMean',
        'CognitiveTermsDailyMean'
    ]

    # Apply rolling average calculation for each metric and create a new column.
    for metric in metrics:
        rolling_avg_column_name = f'{metric}_MA'
        # Calculate mean on the rolling window over the last 7 days
        daily_stats[rolling_avg_column_name] = daily_stats[metric].rolling(window=window_size, min_periods=1).mean()

    return daily_stats
    
def calculate_weekly_rolling_averages(weekly_stats, window_size=4):
    """
    Calculates rolling averages for weekly statistics.

    :param weekly_stats: DataFrame containing weekly statistics.
    :param window_size: The size of the moving window.
    :return: DataFrame with rolling averages appended.
    """

    # Define the metrics we want
    metrics = [
        'SentimentWeeklyMean',
        'FirstPersonSingularWeeklyMean',
        'FirstPersonPluralWeeklyMean',
        'SecondPersonWeeklyMean',
        'CognitiveTermsWeeklyMean'
    ]

    # Apply rolling mean calculation to each metric and create corresponding new columns.
    for metric in metrics:
        rolling_avg_column_name = f'{metric}_MA'
        # Calculate mean on the rolling window over the last month
        weekly_stats[rolling_avg_column_name] = weekly_stats[metric].rolling(window=window_size, min_periods=1).mean()
    
    return weekly_stats

def calculate_daily_z_scores(daily_stats, baseline_stats):
    """
    Calculates z-scores for daily statistics, handling NaN values and zero standard deviations appropriately.

    :param daily_stats: Daily statistics dataframe.
    :param baseline_stats: Baseline statistics dictionary.
    :return: The daily_stats DataFrame updated with z-score columns.
    """

    # Define the features
    features = ['Sentiment', 'FirstPersonSingular', 'FirstPersonPlural', 'SecondPerson', 'CognitiveTerms']

    # Iterate over the features
    for feature in features:
        mean_column = f'{feature}DailyMean'
        z_score_column = f'{feature}DailyZScore'

        # Retrieve baseline mean and standard deviation. Return zero if not found.
        baseline_mean = baseline_stats.get(f'{feature}_mean', 0)
        baseline_std = baseline_stats.get(f'{feature}_std', 0)

        # Handle the edge case where baseline standard deviation is zero.
        if baseline_std == 0:
            daily_stats[z_score_column] = 0
        else:
            # Calculate z-score and fill NaN results, with zeros.
            daily_stats[z_score_column] = (daily_stats[mean_column] - baseline_mean) / baseline_std
            daily_stats[z_score_column].fillna(0, inplace=True)

    return daily_stats

def calculate_weekly_z_scores(weekly_stats, baseline_stats):
    """
    Calculates z-scores for weekly statistics, handling NaN values and zero standard deviations appropriately.

    :param weekly_stats: A DataFrame containing weekly statistics.
    :param baseline_stats: A dictionary with baseline statistics.
    :return: The weekly_stats DataFrame updated with z-score columns.
    """

    # Define the features
    features = ['Sentiment', 'FirstPersonSingular', 'FirstPersonPlural', 'SecondPerson', 'CognitiveTerms']

    # Iterate over features
    for feature in features:
        mean_column = f'{feature}WeeklyMean'
        z_score_column = f'{feature}WeeklyZScore'

        # Retrieve baseline mean and standard deviation. Return zero if not found.
        baseline_mean = baseline_stats.get(f'{feature}_mean', 0)
        baseline_std = baseline_stats.get(f'{feature}_std', 0)

        # Handle the edge case where baseline standard deviation is zero.
        if baseline_std == 0:
            weekly_stats[z_score_column] = 0
        else:
            # Calculate z-score and fill NaN results, with zeros.
            weekly_stats[z_score_column] = (weekly_stats[mean_column] - baseline_mean) / baseline_std
            weekly_stats[z_score_column].fillna(0, inplace=True)

    return weekly_stats

def update_daily_analysis_table(daily_stats, table):
    # Iterate over the daily_stats dataframe and create item to update.
    for index, row in daily_stats.iterrows():
        item = {
            'userid': row['userid'],
            'date': row['just_date'].strftime('%Y-%m-%d'),
            'SentimentDailyMean': str(row['SentimentDailyMean']),
            'SentimentDailyZScore': str(row['SentimentDailyZScore']),
            'FirstPersonSingularDailyMean': str(row['FirstPersonSingularDailyMean']),
            'FirstPersonSingularDailyZScore': str(row['FirstPersonSingularDailyZScore']),
            'FirstPersonPluralDailyMean': str(row['FirstPersonPluralDailyMean']),
            'FirstPersonPluralDailyZScore': str(row['FirstPersonPluralDailyZScore']),
            'SecondPersonDailyMean': str(row['SecondPersonDailyMean']),
            'SecondPersonDailyZScore': str(row['SecondPersonDailyZScore']),
            'CognitiveTermsDailyMean': str(row['CognitiveTermsDailyMean']),
            'CognitiveTermsDailyZScore': str(row['CognitiveTermsDailyZScore']),
            'SentimentDailyMean_MA': str(row['SentimentDailyMean_MA']),
            'FirstPersonSingularDailyMean_MA': str(row['FirstPersonSingularDailyMean_MA']),
            'FirstPersonPluralDailyMean_MA': str(row['FirstPersonPluralDailyMean_MA']),
            'SecondPersonDailyMean_MA': str(row['SecondPersonDailyMean_MA']),
            'CognitiveTermsDailyMean_MA': str(row['CognitiveTermsDailyMean_MA']),
        }

        # Put the item in the analysis table
        table.put_item(Item=item)

def update_weekly_analysis_table(weekly_stats, table):
    # Iterate over the weekly_stats dataframe and create item to update.
    for index, row in weekly_stats.iterrows():
        item = {
            'userid': row['userid'],
            'date': row['year_week'],
            'SentimentWeeklyMean': str(row['SentimentWeeklyMean']),
            'SentimentWeeklyZScore': str(row['SentimentWeeklyZScore']),
            'FirstPersonSingularWeeklyMean': str(row['FirstPersonSingularWeeklyMean']),
            'FirstPersonSingularWeeklyZScore': str(row['FirstPersonSingularWeeklyZScore']),
            'FirstPersonPluralWeeklyMean': str(row['FirstPersonPluralWeeklyMean']),
            'FirstPersonPluralWeeklyZScore': str(row['FirstPersonPluralWeeklyZScore']),
            'SecondPersonWeeklyMean': str(row['SecondPersonWeeklyMean']),
            'SecondPersonWeeklyZScore': str(row['SecondPersonWeeklyZScore']),
            'CognitiveTermsWeeklyMean': str(row['CognitiveTermsWeeklyMean']),
            'CognitiveTermsWeeklyZScore': str(row['CognitiveTermsWeeklyZScore']),
            'SentimentWeeklyMean_MA': str(row['SentimentWeeklyMean_MA']),
            'FirstPersonSingularWeeklyMean_MA': str(row['FirstPersonSingularWeeklyMean_MA']),
            'FirstPersonPluralWeeklyMean_MA': str(row['FirstPersonPluralWeeklyMean_MA']),
            'SecondPersonWeeklyMean_MA': str(row['SecondPersonWeeklyMean_MA']),
            'CognitiveTermsWeeklyMean_MA': str(row['CognitiveTermsWeeklyMean_MA']),
        }

        # Put the item in the analysis table
        table.put_item(Item=item)