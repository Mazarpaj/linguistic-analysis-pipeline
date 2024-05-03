import csv
import boto3
import logging
import os
from datetime import datetime
from io import StringIO

# Retrieve the bucket names from environment variables
raw_bucket_name = os.environ.get('RAW_BUCKET_NAME')
processed_bucket_name = os.environ.get('PROCESSED_BUCKET_NAME')

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize the S3 client
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda handler function.
    Main lambda function, processes CSV files from the raw bucket and writes out to processed bucket.
    """
    # Check if bucket names are configured
    if not raw_bucket_name or not processed_bucket_name:
        logger.error('Bucket names are not configured.')
        return {'statusCode': 500, 'body': 'Configuration error'}

    # Check if records are in event
    if 'Records' not in event:
        logger.error('Invalid event format.')
        return {'statusCode': 400, 'body': 'Invalid event format'}

    try:
        # Process each record in the event
        for record in event['Records']:
            # Skip non-S3 records, as we only want s3 related triggers
            if 's3' not in record or 'object' not in record['s3']:
                logger.warning('Skipping non-S3 record.')
                continue

            # Get key for raw CSV file from record
            csv_file_key = record['s3']['object']['key']
            # Skip non-CSV files
            if not csv_file_key.endswith(".csv"):
                logger.warning(f'Skipped non-CSV file: {csv_file_key}')
                continue

            # Process the CSV file
            process_csv_in_chunks(raw_bucket_name, csv_file_key, processed_bucket_name)

        # Return success message
        logger.info('CSV processing completed successfully')
        return {'statusCode': 200, 'body': 'CSV processing completed successfully'}
    
    # Default error handling
    except Exception as e:
        logger.error('Error processing CSV: %s', str(e))
        return {'statusCode': 500, 'body': 'Error processing CSV'}

def process_csv_in_chunks(bucket_name, csv_file_key, output_bucket_name):
    """
    Processes a CSV file in chunks.
    Reads the raw csv file, processes it, and writes the processed content to the processed bucket.
    """
    try:
        # Get the raw CSV file from the raw bucket
        csv_file = s3_client.get_object(Bucket=bucket_name, Key=csv_file_key)
    
    # If file does not exist
    except s3_client.exceptions.NoSuchKey:
        logger.error(f'File not found: {csv_file_key}')
        return

    # Handle other errors
    except Exception as e:
        logger.error(f'Error getting the file: {str(e)}')
        return

    # Get lines of the raw CSV file
    lines = csv_file['Body'].iter_lines()
    try:
        # Get the headers
        header = next(lines).decode('utf-8')
    
    # Handle iteration error
    except StopIteration:
        logger.error(f'Empty file: {csv_file_key}')
        return

    # Split the headers into fields
    header_fields = header.split(",")

    # Create stringIO object as buffer
    output = StringIO()

    # Initialize a CSV writer with the header fields
    csv_writer = csv.DictWriter(output, fieldnames=header_fields, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writeheader()

    # Process each line
    for line in lines:
        # Obtain line content
        line_content = line.decode('utf-8')

        # Try and read line as dictionary
        try:
            row = next(csv.DictReader(StringIO(line_content), fieldnames=header_fields))
        except StopIteration:
            logger.error('Error reading line.')
            continue

        # Process the rows for the sent emails from the user
        if 'activity' in row and row['activity'].strip().lower() == 'send':
            try:
                # Handle date formatting
                row['date'] = parse_date(row['date'])
            except ValueError as e:
                logger.warning(f'Date format error: {str(e)}')
                continue

            # Write processed row content
            csv_writer.writerow(row)

    # Save the processed CSV file to the processed bucket
    save_processed_content(output.getvalue(), f'processed_{csv_file_key}', output_bucket_name)

def parse_date(date_str):
    """
    Parses a date string into an ISO format date string.
    Tries the differing date formats found in the CERT dataset file.
    """
    # Declares the date formats and iterates through them
    for fmt in ('%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M'):
        try:
            # Return formatted date in ISO format
            return datetime.strptime(date_str, fmt).isoformat()
        except ValueError:
            continue
    raise ValueError('no valid date format found')

def save_processed_content(processed_content, csv_file_key, bucket_name):
    """
    Saves the processed CSV to the processed bucket.
    """
    try:
        # Put processed CSV into processed bucket.
        s3_client.put_object(Bucket=bucket_name, Key=csv_file_key, Body=processed_content)
    except Exception as e:
        logger.error(f'Error saving processed file: {str(e)}')
        return
    # Log
    logger.info(f'Processed CSV written to {csv_file_key} in bucket {bucket_name}')
