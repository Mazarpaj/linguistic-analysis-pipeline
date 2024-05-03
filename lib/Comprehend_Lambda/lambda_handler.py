import boto3
import csv
import os
from io import StringIO
import json
from decimal import Decimal
import logging

# Initialize AWS clients
s3_client = boto3.client('s3')
comprehend_client = boto3.client('comprehend')
dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')

# Retrieve the environment variables
processed_bucket_name = os.environ['PROCESSED_BUCKET_NAME']
results_table_name = os.environ['RESULTS_TABLE_NAME']
user_table_name = os.environ['USER_TABLE_NAME']
nltk_lambda_name = os.environ['NLTK_LAMBDA_NAME']

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants
BATCH_SIZE = 25  # Maximum batch size for BatchDetectSentiment
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for sentiment results
USERS_TO_PROCESS = {'HBP1076', 'OFS0030', 'RCG0584','BDP0096'}
# 
def lambda_handler(event, context):
    """
    Main lambda function.

    Args:
        event: Event dictionary containing information about S3 object.
        context: Lambda context.

    Returns:
        Response object.
    """
    try:
        for record in event['Records']:
            # Extract bucket name and processed CSV file key from records.
            bucket_name = record['s3']['bucket']['name']
            csv_file_key = record['s3']['object']['key']

            # Check if the file is from the processed data bucket
            if bucket_name == processed_bucket_name and csv_file_key.endswith(".csv"):
                # Read and process the CSV file.
                csv_file = s3_client.get_object(Bucket=bucket_name, Key=csv_file_key)
                # Using UTF-8-SIG as format seemed to require it.
                csv_content = csv_file['Body'].read().decode('utf-8-sig')

                process_csv_content(csv_content, csv_file_key)

        # Log success
        logger.info('Comprehend analysis completed successfully.')
        return {
            'statusCode': 200,
            'body': 'Comprehend analysis completed successfully'
        }
    
    # Error handling
    except Exception as e:
        logger.error(f'Error processing CSV: {str(e)}')
        return {
            'statusCode': 500,
            'body': 'Error processing CSV'
        }
        
def is_processed(user_id, email_id, table):
    """
    Check if the email is already processed.

    Args:
        user_id: User ID.
        email_id: Email ID.
        table: DynamoDB table object.

    Returns:
        True if processed, otherwise false.
    """
    try:
        # Retrieve email record from results table
        response = table.get_item(Key={
            'userid': user_id,
            'emailid': email_id
        })
        # Assume record having date indicates it has been processed.
        return 'Item' in response and 'date' in response['Item']
    except Exception as e:
        logger.error(f'Error checking processed status for {user_id} {email_id}: {str(e)}')
        return False  # Default to False so that record is processed


def process_csv_content(csv_content, csv_file_key):
    """
    Process email records, call comprehend to obtain sentiment and update tables.

    Args:
        csv_content: The content of processed CSV file.
        csv_file_key: The S3 key for processed CSV file.
    """
    # Read CSV content using DictReader
    csv_reader = csv.DictReader(StringIO(csv_content))

    # Set up dynamodb table objects
    results_table = dynamodb.Table(results_table_name)
    user_table = dynamodb.Table(user_table_name)

    # Initialise batches to accumulate elements for batch processing
    text_batch = []
    email_id_batch = []
    user_id_batch = []
    date_batch = []

    # Iterate through email records
    for row in csv_reader:

        # Obtain attributes from email record
        email_id = row['id']
        content = row['content']
        user_id = row['user']
        date = row['date']

        # If we don't want to process the user then skip
        if user_id not in USERS_TO_PROCESS:
            continue
        
        # Check if record is already processed
        if is_processed(user_id, email_id, results_table):
            continue
        
        # If there is email content, e.g. check for null field to avoid inefficiency, then append to batch
        if content:
            text_batch.append(content)
            email_id_batch.append(email_id)
            user_id_batch.append(user_id)
            date_batch.append(date)

        # Check if batch is full / Process in maximum Comprehend batch size
        if len(text_batch) == BATCH_SIZE:
            # Obtain sentiment analysis results
            sentiment_results = analyze_batch(text_batch)
            # Update results table
            update_results_table(user_id_batch, email_id_batch, date_batch, sentiment_results, results_table)
            # Reset the batches for the next batch
            text_batch, email_id_batch, user_id_batch, date_batch = [], [], [], []

        # Update the user table. NOT CURRENTLY USED
        update_user_table(user_id, user_table)

    # Process remaining records that did not fit in batches
    if text_batch:
        sentiment_results = analyze_batch(text_batch)
        update_results_table(user_id_batch, email_id_batch, date_batch, sentiment_results, results_table)
    
    # Directly invoke nltk lambda as AWS free tier has limits on data stream triggers.
    invoke_nltk_lambda(csv_file_key, processed_bucket_name)

def invoke_nltk_lambda(csv_file_key, processed_bucket_name):
    """
    Invoke the NLTK lambda.

    Args:
        csv_file_key: S3 key of processed CSV file.
        processed_bucket_name: Name of the processed data S3 bucket.

    Returns:
        Lambda response.
    """
    # Pass the csv file and bucket info
    payload = {
        'csv_file_key': csv_file_key,
        'bucket_name': processed_bucket_name
    }
    # Invoke the lambda
    response = lambda_client.invoke(
        FunctionName=nltk_lambda_name,
        InvocationType='Event',
        Payload=json.dumps(payload),
    )
    return response

def analyze_batch(text_batch):
    """
    Analyze batch of content using Comprehend.

    Args:
        text_batch: List of content text to analyse.

    Returns:
        The result list from the batch sentiment analysis.
    """
    # Call Comprehend on the batch, defining the language as english to save latency and cost
    response = comprehend_client.batch_detect_sentiment(TextList=text_batch, LanguageCode='en')
    return response['ResultList']

def update_results_table(user_id_batch, email_id_batch, date_batch, sentiment_results, table):
    """
    Update the results table with sentiment analysis data.

    Args:
        user_id_batch: List of the User IDs.
        email_id_batch: List of the Email IDs.
        date_batch: List of the email dates.
        sentiment_results: List of the email sentiment results, including type and confidence.
        table: DynamoDB results table.
    """
    # Using batch writer to reduce calls to DynamoDB
    with table.batch_writer() as batch:
        # Iterate through enumerated sentiment results
        for i, sentiment_data in enumerate(sentiment_results):

            # Construct the item to write
            user_id = user_id_batch[i]
            email_id = email_id_batch[i]
            date = date_batch[i]
            sentiment_scores = sentiment_data['SentimentScore']
            # Obtain the sentiment type with the highest confidence.
            max_sentiment, max_score = max(sentiment_scores.items(), key=lambda x: x[1])

            # Always create the base item with or without sentiment data based on the confidence threshold. Ensures data integrity through the pipeline and avoids errors further on.
            item = {
                'userid': user_id,
                'emailid': email_id,
                'date': date,
                'pronouns': {},
                'cognitiveterms': {}
            }

            # Only add sentiment data if the confidence score is above the threshold
            if max_score >= CONFIDENCE_THRESHOLD:
                max_score_dec = Decimal(str(max_score))
                item['sentiment'] = max_sentiment
                item['confidence'] = max_score_dec

            # Add the item to the batch to be written to the table
            batch.put_item(Item=item)

def update_user_table(user_id, table):
    """
    Add user to user table.

    Args:
        user_id: User ID.
        table: DynamoDB table for user table.
    """
    # Check if user exists in table
    response = table.get_item(Key={'userid': user_id})

    # If user does not exist, add user to table
    if 'Item' not in response:
        table.put_item(Item={'userid': user_id})