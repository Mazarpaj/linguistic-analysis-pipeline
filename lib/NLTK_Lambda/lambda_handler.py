import os
import boto3
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from collections import Counter
from decimal import Decimal
from io import StringIO
import csv
import logging

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')

# Retrieve the environment variables
processed_bucket_name = os.environ['PROCESSED_BUCKET_NAME']
dynamodb_table_name = os.environ['RESULTS_TABLE_NAME']
anomaly_lambda_name = os.environ['ANOMALY_LAMBDA_NAME']

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# NLTK data path setup within lambda
nltk.data.path.append("/opt/nltk_data")

# CONSTANTS
USERS_TO_PROCESS = {'HBP1076', 'BDP0096', 'OFS0030', 'RCG0584'}

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
        # Get bucket name and preprocessing CSV file key
        bucket_name = event.get('bucket_name')
        csv_file_key = event.get('csv_file_key')

        # Check if the file is from the processed data bucket and is a CSV
        if bucket_name == processed_bucket_name and csv_file_key.endswith(".csv"):
            csv_file = s3_client.get_object(Bucket=bucket_name, Key=csv_file_key)
            # Extract CSV content / email records
            csv_content = csv_file['Body'].read().decode('utf-8-sig')

            # Process the email records
            process_csv_content(csv_content)

        # Log success
        logger.info('NLTK analysis completed successfully.')
        return {
            'statusCode': 200,
            'body': 'NLTK analysis completed successfully'
        }
    
    # Log errors
    except Exception as e:
        logger.error(f'Error processing CSV: {str(e)}')
        return {
            'statusCode': 500,
            'body': 'Error processing CSV'
        }
        
def should_process(user_id, email_id, table):
    """
    Check if the provided email_id has been processed already.

    Args:
    email_id (str): Email ID to check.
    table (boto3.resources.factory.dynamodb.Table): The DynamoDB table.

    Returns:
    bool: True if the record should be processed, otherwise False.
    """
    try:
        # Obtain email record from results table
        response = table.get_item(Key={
            'userid': user_id,
            'emailid': email_id
        })
        # Check if the record exists and has 'pronouns' or 'cognitiveterms' attributes
        item = response.get('Item', {})
        # # Check for missing attributes or attributes with empty maps
        # return 'pronouns' not in item or not item['pronouns'] or 'cognitiveterms' not in item or not item['cognitiveterms']
        # Check for both 'pronouns' and 'cognitiveterms' attributes being either not present
        # or present with empty maps
        return ('pronouns' not in item or not item['pronouns']) and ('cognitiveterms' not in item or not item['cognitiveterms'])
    except Exception as e:
        logger.error(f"Error retrieving item from DynamoDB: {str(e)}")
        # If there's an error accessing the table, we default to processing the record
        return True
    
def process_csv_content(csv_content):
    """
    Process CSV content to obtain cognitive terms and pronouns using NLTK.

    Args:
        csv_content: The content of the CSV file to be processed.
    """
    # Read CSV content using DictReader
    csv_reader = csv.DictReader(StringIO(csv_content))
    # Set up dynamodb table objects
    table = dynamodb.Table(dynamodb_table_name)

    # Base terms for cognitive analysis from LIWC
    base_terms = [
        'cause', 'know', 'ought', 'think', 'consider', 'because', 
        'effect', 'hence', 'should', 'would', 'could', 'maybe', 
        'perhaps', 'guess', 'always', 'never', 'block', 'constrain', 
        'with', 'and', 'include', 'but', 'except', 'without'
    ]

    # Use wordnet's sysnet functionality to obtain a wider set of cognitive terms
    cognitive_terms = sysnet_terms(base_terms)

    # Iterate through email records
    for row in csv_reader:

        # If we don't want to process the user then skip
        if row['user'] not in USERS_TO_PROCESS:
            continue

        # Before calculating frequencies, check if this row has been processed
        if not should_process(row['user'], row['id'], table):
            # Skip if already processed to save on resources and latency
            logger.info(f"Skipping already processed record {row['id']}")
            continue

        # Obtain attributes from email record
        content = row['content']
        email_id = row['id']
        user_id = row['user']

        # Calculate frequencies
        pronoun_freq, cognitive_term_freq = calculate_frequencies(content, cognitive_terms)

        # Only update the table if there are non-empty frequency results
        if pronoun_freq or cognitive_term_freq:
            update_table(table, user_id, email_id, pronoun_freq, cognitive_term_freq)
        else:
            print(f"No frequencies to update for user {user_id} and email {email_id}")
    
    # Directly invoke anomaly lambda as AWS free tier has limits on data stream triggers.
    invoke_anomaly_lambda()


def invoke_anomaly_lambda():
    """
    Invoke the anomaly lambda.

    Returns:
        Lambda response.
    """
    response = lambda_client.invoke(
        FunctionName=anomaly_lambda_name,
        InvocationType='Event',
    )
    return response


def calculate_frequencies(content, cognitive_terms):
    """
    Calculates the frequencies of pronouns and cognitive terms in the email.

    Args:
        content: The email content to analyse.
        cognitive_terms: The expanded set of cognitive terms to check for.

    Returns:
        Two dictionaries containing the frequencies of pronouns and cognitive terms respectively.
    """

    # Tokenize content for use with NLTK
    tokens = word_tokenize(content)

    # Tag the tokens using the part-of-speech tags
    tagged_tokens = pos_tag(tokens)

    # Ensure normalisation as converting terms to lower case before checking.
    # Extract the tagged pronouns from the tagged tokens
    pronouns = [word.lower() for word, tag in tagged_tokens if tag in ['PRP', 'PRP$']]

    # Count the frequency of the pronouns
    pronoun_freq = Counter(pronouns)

    # Extract the tagged cognitive terms from the tagged tokens
    cognitive_term_tokens = [token.lower() for token in tokens if token.lower() in cognitive_terms]

    # Count the frequency of the cognitive terms
    cognitive_term_freq = Counter(cognitive_term_tokens)

    # Return the dictionaries of the terms and their associated frequencies
    return dict(pronoun_freq), dict(cognitive_term_freq)

def sysnet_terms(base_terms):
    """
    Expands base terms into a set of related terms using WordNet synsets.

    Args:
        base_terms: Set of strings to represent the base terms.

    Returns:
        Set of strings representing the expanded terms.
    """

    # Initialise set for expanded terms
    expanded_terms = set()

    # Iterate through the base terms and obtain expanded terms for each
    for term in base_terms:
        # Get synsets for the term
        for syn in wn.synsets(term):
            # Add all words from each sysnet
            expanded_terms.update(lemma.name().replace('_', ' ') for lemma in syn.lemmas())

    # Return the expanded terms
    return expanded_terms

def update_table(table, user_id, email_id, pronoun_freq, cognitive_term_freq):
    """
    Update DynamoDB table by inserting dictionaries as map attributes.

    :param table: The DynamoDB table object to update.
    :param user_id: The user ID associated with the record.
    :param email_id: The email ID associated with the record.
    :param pronoun_freq: A dictionary of pronoun frequencies.
    :param cognitive_term_freq: A dictionary of cognitive term frequencies.
    """

    # DynamoDB does not allow empty maps or lists, so check before update
    if not pronoun_freq and not cognitive_term_freq:
        print("Both pronoun and cognitive term frequencies are empty. No update performed.")
        return
    
    # Initialise expressions for update
    expression_attribute_values = {}
    update_expression_parts = []

    # Add entire dictionary for pronouns and cognitive terms if they are not empty
    # Ensure frequencies are converted to a decimal for correct storage in DynamoDB and data integrity throughout the pipeline.
    if pronoun_freq:
        expression_attribute_values[':pronouns'] = {k: Decimal(str(v)) for k, v in pronoun_freq.items()}
        update_expression_parts.append('pronouns = :pronouns')
    
    if cognitive_term_freq:
        expression_attribute_values[':cognitiveterms'] = {k: Decimal(str(v)) for k, v in cognitive_term_freq.items()}
        update_expression_parts.append('cognitiveterms = :cognitiveterms')

    # Construct the update expression from the parts
    update_expression = 'SET ' + ', '.join(update_expression_parts)

    # Prepare the update parameters
    update_params = {
        "Key": {
            'userid': user_id,
            'emailid': email_id
        },
        "UpdateExpression": update_expression,
        "ExpressionAttributeValues": expression_attribute_values
    }

    try:
        # Update the record
        response = table.update_item(**update_params)
        print(f"Successfully updated the table for user {user_id} and email {email_id}.")
    except Exception as e:
        logger.error(f"Error updating table for user {user_id} and email {email_id}: {str(e)}")
        raise