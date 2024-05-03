import boto3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from io import BytesIO
import os
import logging

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Environment variables
analysis_table_name = os.environ.get('ANALYSIS_TABLE_NAME')
output_bucket_name = os.environ.get('OUTPUT_BUCKET_NAME')
os.environ['MPLCONFIGDIR'] = '/tmp'

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants
USERS_TO_PROCESS = {'HBP1076', 'OFS0030', 'RCG0584', 'BDP0096'}

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
        # Fetching and preprocessing data.
        df = fetch_all_data_from_dynamodb()
        df_baseline = preprocess_baseline(df)
        df_weekly = preprocess_data(df)

        # Plotting and uploading.
        plot_pronouns_weekly(df_weekly, output_bucket_name)
        plot_pronoun_ratios(df_weekly, output_bucket_name, df_baseline)
        plot_pronoun_zscore(df_weekly, output_bucket_name)
        plot_cognitive_terms_weekly(df_weekly, output_bucket_name, df_baseline)
        plot_sentiment_weekly(df_weekly, output_bucket_name, df_baseline)
        plot_cognitive_term_zscore(df_weekly, output_bucket_name)
        plot_sentiment_zscore(df_weekly, output_bucket_name)

    except Exception as e:
        # Log the error and return a 500 status code for failure.
        logger.error(f"An error occurred: {str(e)}")
        return {
            'statusCode': 500,
            'body': 'An error occurred while generating plots.'
        }

    # Return a success response on success.
    return {
        'statusCode': 200,
        'body': 'Successfully generated and uploaded plots.'
    }

def fetch_all_data_from_dynamodb():
    """
    Fetches all data from analysis table. FUNCTION SHOULD ONLY BE USED IN TESTING. SCAN SHOULD BE CHANGED TO MORE SPECIFIC QUERY FOR PRODUCTION.

    :return: A pandas DataFrame containing all items from the analysis table.
    """
    try:
        # Set up analysis table instance
        table = dynamodb.Table(analysis_table_name)
        # Scan to get all items
        response = table.scan()
        # Extract records from response
        items = response['Items']
        
        # Handle pagination if the response is truncated.
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response['Items'])

        # Return the records as a dataframe
        return pd.DataFrame(items)

    except Exception as e:
        logger.error(f"Failed to fetch data from DynamoDB: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on failure.

def preprocess_data(df):
    """
    Processes user data for weekly plot generation, adjusting date formats.

    :param df: DataFrame containing user data.
    :return: DataFrame with processed weekly data.
    """

    # Check if input dataframe is empty
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return df

    try:
        # Process 'date' to ensure it's in datetime format for weekly data.
        # %Y' represents the year, 'W%W' represents the week of the year, and '%w' represents the day of the week (where Monday is 0 and Sunday is 6)
        # Will be set to NaT (Not a Time) format if error occurs using coerce.
        # -1 is added to ensure date is set to first day of the month
        df['date'] = pd.to_datetime(df['date'].astype(str) + '-1', format='%Y-W%W-%w', errors='coerce')

        # Drop rows with invalid 'date' conversions.
        df.dropna(subset=['date'], inplace=True)

        # Filter the DataFrame to include only the specified users.
        df = df[df['userid'].isin(USERS_TO_PROCESS)]

        return df

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on failure.
        
def preprocess_baseline(df):
    """
    Processes user data to extract baseline records.

    :param df: DataFrame containing user data.
    :return: DataFrame with processed weekly and baseline data.
    """
    
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return df

    try:
        # Separate baseline records from weekly records
        baseline_df = df[df['date'] == 'baseline']

        return baseline_df

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on failure.


def plot_pronouns_weekly(df, output_bucket_name):
    """
    Generates and uploads weekly pronoun usage plots for user.

    :param df: DataFrame containing pronoun usage data.
    :param output_bucket_name: Name of output bucket.
    """

    # Set the seaborn style
    sns.set_theme(style="darkgrid")

    for user_id in USERS_TO_PROCESS:
        # Creates the dataframe for the user data and sorts by date.
        user_df = df[df['userid'] == user_id].sort_values(by='date')
        
        # Rounding the metrics to 2 dp for better graph aesthetic
        for metric in ['FirstPersonPluralWeeklyMean_MA', 'FirstPersonSingularWeeklyMean_MA', 'SecondPersonWeeklyMean_MA']:
            user_df[metric] = user_df[metric].round(2)

        # Melt the DataFrame for Seaborn
        # Basically combining the pronoun columns into a single column so that we can display the total pronouns Mean_MA.
        # date is the identifying column.
        user_df_melted = user_df.melt(id_vars=['date'], 
                                      value_vars=['FirstPersonPluralWeeklyMean_MA', 
                                                  'FirstPersonSingularWeeklyMean_MA', 
                                                  'SecondPersonWeeklyMean_MA'],
                                      var_name='Metric', value_name='Value')

        # Initialize plot
        plt.figure(figsize=(10, 6))

        # Plot data using Seaborn, date on the x axis, and the total pronoun Mean_MA on the y-axis.
        sns.lineplot(x='date', y='Value', hue='Metric', data=user_df_melted, marker='o')

        # Check that there aren't any NaN values after melting
        if user_df_melted['Value'].notna().any():
            # Setting plot title and labels
            plt.title(f"Weekly Pronoun Usage for User {user_id}")
            plt.xlabel('Date')
            plt.ylabel('Frequency')
            plt.legend(title='Metric', labels=[metric.replace('WeeklyMean_MA', '') for metric in 
                                              ['FirstPersonPlural', 'FirstPersonSingular', 'SecondPerson']])
            
        # Attempt to save and upload the graph to the output bucket.
        try:
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, output_bucket_name, f"{user_id}/pronoun_usage_weekly.png")
            buffer.close()
        except Exception as e:
            logger.error(f"Error generating or uploading plot for user {user_id}: {e}")
        finally:
            # Close figure to free resources
            plt.close()

        #     # Saving to BytesIO object and uploading to S3
        #     output_path = f"{user_id}/pronoun_usage_weekly.png"
        #     buffer = BytesIO()
        #     plt.savefig(buffer, format='png')
        #     buffer.seek(0)
        #     s3_client.upload_fileobj(buffer, output_bucket_name, output_path)
        # else:
        #     print(f"No weekly pronoun data available to plot for user {user_id}")
        
        # plt.close()
        
def plot_pronoun_ratios(df, output_bucket_name, df_baseline):
    """
    Generates and uploads plots of pronoun usage ratios over time for users.

    Pronoun ratio is calculated as (FirstPersonSingular + SecondPerson) / FirstPersonPlural.
    Ratios are plotted over time for each user, and the plots are uploaded to an S3 bucket.

    :param df: DataFrame containing the pronoun usage data.
    :param output_bucket_name: Name of output bucket.
    """
    sns.set_theme(style="darkgrid")

    for user_id in USERS_TO_PROCESS:
        # Sorted by date
        user_df = df[df['userid'] == user_id].sort_values(by='date')
        user_baseline_df = df_baseline[df_baseline['userid'] == user_id]
        
        if not user_baseline_df.empty:
            baseline_ratio = (
                float(user_baseline_df['FirstPersonSingular_mean']) +
                float(user_baseline_df['SecondPerson_mean'])
            ) / float(user_baseline_df['FirstPersonPlural_mean'])

        # Ensure numeric conversion; handle potential non-numeric types, by setting to NaN
        cols_to_convert = ['FirstPersonPluralWeeklyMean_MA', 'FirstPersonSingularWeeklyMean_MA', 'SecondPersonWeeklyMean_MA']
        user_df[cols_to_convert] = user_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        # Calculate pronoun ratio and round it to 2 dp.
        # Drop the NaN columns.
        user_df = user_df.dropna(subset=cols_to_convert)
        user_df['Pronoun_Ratio'] = (user_df['FirstPersonSingularWeeklyMean_MA'] + user_df['SecondPersonWeeklyMean_MA']) / user_df['FirstPersonPluralWeeklyMean_MA']
        
        user_df['Pronoun_Ratio'] = user_df['Pronoun_Ratio'].round(2)

        # Check that there aren't any NaN values after melting
        if user_df['Pronoun_Ratio'].notna().any():
            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(data=user_df, x='date', y='Pronoun_Ratio', marker='o', label='(FirstPersonSingular + SecondPerson) / FirstPersonPlural')
            if not user_baseline_df.empty:
                plt.axhline(y=baseline_ratio, color='red', linestyle='--', label='Baseline Ratio')
            plt.title(f"Pronoun Ratio over Time for User {user_id}")
            plt.xlabel('Date')
            plt.ylabel('Ratio')
        
            # Set major ticks to the first of each month and format them to show only the month and year
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
            # Ensure that each tick corresponds to a date and is readable by rotating 45 degrees
            plt.xticks(rotation=45)
            plt.tight_layout()
        
            plt.legend()

            # Attempt to save and upload the graph to the output bucket.
            try:
                output_path = f"{user_id}/pronoun_ratio_weekly.png"
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                # Move buffer pointer back to start
                buffer.seek(0)
                s3_client.upload_fileobj(buffer, output_bucket_name, output_path)
            except Exception as e:
                logger.error(f"Error in generating or uploading plot for user {user_id}: {e}")
            finally:
                # Close figure and buffer to free resources
                plt.close()
                buffer.close()
        else:
            logger.info(f"No sufficient data for pronoun ratio calculation for user {user_id}")

        #     # Save to BytesIO object and upload
        #     output_path = f"{user_id}/pronoun_ratio_weekly.png"
        #     buffer = BytesIO()
        #     plt.savefig(buffer, format='png')
        #     buffer.seek(0)
        #     s3_client.upload_fileobj(buffer, output_bucket_name, output_path)
        #     plt.close()
        # else:
        #     print(f"No sufficient data for pronoun ratio calculation for user {user_id}")

def plot_pronoun_zscore(df, output_bucket_name):
    """
    Generates and uploads z-score plots for weekly pronoun usage.

    Creates three graph figure for each user, plotting the z-scores for each pronoun type.

    :param df: A pandas DataFrame containing z-score data for pronoun metrics.
    :param output_bucket_name: The name of the S3 bucket for storing plots.
    """
    sns.set_theme(style="darkgrid")

    for user_id in USERS_TO_PROCESS:
        user_df = df[df['userid'] == user_id].sort_values(by='date')

        # Ensure numeric conversion; handle potential non-numeric types, by setting to NaN
        cols_to_convert = ['FirstPersonPluralWeeklyZScore', 'FirstPersonSingularWeeklyZScore', 'SecondPersonWeeklyZScore']
        user_df[cols_to_convert] = user_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values
        user_df = user_df.dropna(subset=cols_to_convert)

        if not user_df.empty:
            fig, axs = plt.subplots(3, 1, figsize=(10, 16), sharex=True)

            # Plot FirstPersonPluralWeeklyZScore
            ax = axs[0]
            sns.lineplot(data=user_df, x='date', y='FirstPersonPluralWeeklyZScore', marker='o', ax=ax)
            ax.set_title(f"FirstPersonPlural Weekly Z-Score for User {user_id}")
            ax.set_ylabel('Z-Score')

            # Plot FirstPersonSingularWeeklyZScore
            ax = axs[1]
            sns.lineplot(data=user_df, x='date', y='FirstPersonSingularWeeklyZScore', marker='o', ax=ax)
            ax.set_title(f"FirstPersonSingular Weekly Z-Score for User {user_id}")
            ax.set_ylabel('Z-Score')

            # Plot SecondPersonWeeklyZScore
            ax = axs[2]
            sns.lineplot(data=user_df, x='date', y='SecondPersonWeeklyZScore', marker='o', ax=ax)
            ax.set_title(f"SecondPerson Weekly Z-Score for User {user_id}")
            ax.set_xlabel('Date')
            ax.set_ylabel('Z-Score')

            # Set major ticks to the first of each month and format them to show only the month and year
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # Ensure that each tick corresponds to a date and is readable
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Attempt to save and upload the graph to the output bucket.
            try:
                output_path = f"{user_id}/pronoun_zscore_weekly.png"
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                # Move buffer pointer back to start
                buffer.seek(0)
                s3_client.upload_fileobj(buffer, output_bucket_name, output_path)
            except Exception as e:
                logger.error(f"Error generating or uploading z-score plot for user {user_id}: {e}")
            finally:
                # Close figure and buffer to free resources
                buffer.close()
                plt.close()
        else:
            logger.info(f"No sufficient data for pronoun z-score plotting for user {user_id}")

def plot_cognitive_terms_weekly(df, output_bucket_name, df_baseline):
    """
    Generates and uploads plots representing the weekly moving average of cognitive terms usage.

    :param df: DataFrame containing the data for cognitive terms usage.
    :param output_bucket_name: Name of output bucket.
    """
    sns.set_theme(style="darkgrid")

    for user_id in USERS_TO_PROCESS:
        # Sort by date
        user_df = df[df['userid'] == user_id].sort_values(by='date')
        user_baseline_df = df_baseline[df_baseline['userid'] == user_id]
        baseline_mean = float(user_baseline_df['CognitiveTerms_mean'])
        
        # Ensure numeric conversion; handle potential non-numeric types, by setting to NaN, round to 2 dp as well
        user_df['CognitiveTermsWeeklyMean_MA'] = pd.to_numeric(user_df['CognitiveTermsWeeklyMean_MA'], errors='coerce')
        user_df['CognitiveTermsWeeklyMean_MA'] = user_df['CognitiveTermsWeeklyMean_MA'].round(2)

        # Check that there aren't any NaN values
        if user_df['CognitiveTermsWeeklyMean_MA'].notna().any():
            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(data=user_df, x='date', y='CognitiveTermsWeeklyMean_MA', marker='o', label='Cognitive Terms MA')
            if not user_baseline_df.empty:
                plt.axhline(y=baseline_mean, color='red', linestyle='--', label='Baseline Mean')
            plt.title(f"Cognitive Terms MA over Time for User {user_id}")
            plt.xlabel('Date')
            plt.ylabel('Moving Average')

            # Set major ticks to the first of each month and format them to show only the month and year
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # Ensure that each tick corresponds to a date and is readable
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.legend()
            
            # Attempt to save and upload the graph to the output bucket.
            try:
                output_path = f"{user_id}/cognitive_terms_weekly.png"
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                # Move buffer pointer back to start
                buffer.seek(0)
                s3_client.upload_fileobj(buffer, output_bucket_name, output_path)
            except Exception as e:
                logger.error(f"Error in generating or uploading cognitive terms plot for user {user_id}: {e}")
            finally:
                # Close figure and buffer to free resources
                plt.close()
                buffer.close()
        else:
            logger.info(f"No weekly cognitive terms MA data available to plot for user {user_id}")
            
def plot_cognitive_term_zscore(df, output_bucket_name):
    """
    Generates and uploads plots of the weekly z-score for cognitive terms usage.

    :param df: DataFrame containing the cognitive terms data along with z-scores.
    :param output_bucket_name: Name of output bucket.
    """
    sns.set_theme(style="darkgrid")

    for user_id in USERS_TO_PROCESS:
        # Sort by date
        user_df = df[df['userid'] == user_id].sort_values(by='date')

        # Ensure numeric conversion; handle potential non-numeric types, by setting to NaN
        cols_to_convert = ['CognitiveTermsWeeklyZScore']
        user_df[cols_to_convert] = user_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values
        user_df = user_df.dropna(subset=cols_to_convert)

        if not user_df.empty:
            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(data=user_df, x='date', y='CognitiveTermsWeeklyZScore', marker='o')
            plt.title(f"Cognitive Terms Weekly Z-Score for User {user_id}")
            plt.xlabel('Date')
            plt.ylabel('Z-Score')

            # Set major ticks to the first of each month and format them to show only the month and year
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # Ensure that each tick corresponds to a date and is readable
            plt.xticks(rotation=45)
            plt.tight_layout()

            try:
                output_path = f"{user_id}/cognitive_terms_zscore_weekly.png"
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                s3_client.upload_fileobj(buffer, output_bucket_name, output_path)
            except Exception as e:
                logger.error(f"Error in generating or uploading cognitive terms z-score plot for user {user_id}: {e}")
            finally:
                plt.close()
                buffer.close()
        else:
            logger.info(f"No sufficient data for cognitive terms z-score plotting for user {user_id}")
            
def plot_sentiment_weekly(df, output_bucket_name, df_baseline):
    """
    Generates and uploads sentiment weekly moving average plots.

    :param df: DataFrame containing sentiment data.
    :param output_bucket_name: Name of output bucket.
    """
    sns.set_theme(style="darkgrid")

    for user_id in USERS_TO_PROCESS:
        user_df = df[df['userid'] == user_id].sort_values(by='date')
        user_baseline_df = df_baseline[df_baseline['userid'] == user_id]
        baseline_mean = float(user_baseline_df['Sentiment_mean'])

        # Ensure numeric conversion; handle potential non-numeric types
        user_df['SentimentWeeklyMean_MA'] = pd.to_numeric(user_df['SentimentWeeklyMean_MA'], errors='coerce')
        user_df['SentimentWeeklyMean_MA'] = user_df['SentimentWeeklyMean_MA'].round(2)

        # Find the maximum absolute value for setting symmetrical y-axis limits
        max_abs_sentiment = user_df['SentimentWeeklyMean_MA'].abs().max() or 1

        if user_df['SentimentWeeklyMean_MA'].notna().any():
            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(data=user_df, x='date', y='SentimentWeeklyMean_MA', marker='o', label='Sentiment MA')
            if not user_baseline_df.empty:
                plt.axhline(y=baseline_mean, color='red', linestyle='--', label='Baseline Mean')
            plt.title(f"Weekly Sentiment MA for User {user_id}")
            plt.xlabel('Date')
            plt.ylabel('Moving Average')

            # Set the y-axis limits to be symmetrical based on the maximum absolute value found
            plt.ylim([-max_abs_sentiment * 1.1, max_abs_sentiment * 1.1])  
            # Slightly extend the limit for visual clarity.

            # Set major ticks to the first of each month and format them to show only the month and year
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # Ensure that each tick corresponds to a date and is readable
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.legend()
            
            try:
                output_path = f"{user_id}/sentiment_weekly.png"
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                s3_client.upload_fileobj(buffer, output_bucket_name, output_path)
            except Exception as e:
                logger.error(f"Error in generating or uploading sentiment plot for user {user_id}: {e}")
            finally:
                buffer.close()
                plt.close()
        else:
            logger.info(f"No weekly sentiment MA data available to plot for user {user_id}")
            
def plot_sentiment_zscore(df, output_bucket_name):
    """
    Generates and uploads plots of the weekly sentiment z-score.

    :param df: DataFrame containing sentiment z-score data.
    :param output_bucket_name: Name of output bucket.
    """
    sns.set_theme(style="darkgrid")

    for user_id in USERS_TO_PROCESS:
        user_df = df[df['userid'] == user_id].sort_values(by='date')

        # Ensure numeric conversion; handle potential non-numeric types
        cols_to_convert = ['SentimentWeeklyZScore']
        user_df[cols_to_convert] = user_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values
        user_df = user_df.dropna(subset=cols_to_convert)

        if not user_df.empty:
            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(data=user_df, x='date', y='SentimentWeeklyZScore', marker='o')
            plt.title(f"Sentiment Weekly Z-Score for User {user_id}")
            plt.xlabel('Date')
            plt.ylabel('Z-Score')

            # Set major ticks to the first of each month and format them to show only the month and year
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # Ensure that each tick corresponds to a date and is readable
            plt.xticks(rotation=45)
            plt.tight_layout()

            try:
                output_path = f"{user_id}/sentiment_zscore_weekly.png"
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                s3_client.upload_fileobj(buffer, output_bucket_name, output_path)
            except Exception as e:
                logger.error(f"Error in generating or uploading sentiment z-score plot for user {user_id}: {e}")
            finally:
                buffer.close()
                plt.close()
        else:
            logger.info(f"No sufficient data for sentiment z-score plotting for user {user_id}")

