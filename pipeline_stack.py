import aws_cdk as cdk
from constructs import Construct
from aws_cdk import (
    aws_s3 as s3,
    aws_dynamodb as ddb,
    aws_lambda as lambda_,
    aws_iam as iam
)

class PipelineStack(cdk.Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Define the S3 buckets
        self.malicious_bucket = s3.Bucket(self, "MaliciousBucket")
        self.raw_bucket = s3.Bucket(self, "RawBucket")
        self.processed_bucket = s3.Bucket(self, "ProcessedBucket")
        self.output_bucket = s3.Bucket(self, "OutputBucket")

        # Define the DynamoDB table
        self.results_table = ddb.Table(
            self, "ResultsTable",
            partition_key=ddb.Attribute(name="userid", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="emailid", type=ddb.AttributeType.STRING),
        )

        # Adding a Global Secondary Index for querying by user and date
        self.results_table.add_global_secondary_index(
            index_name="UserDateIndex",
            partition_key=ddb.Attribute(name="userid", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="date", type=ddb.AttributeType.STRING),
            projection_type=ddb.ProjectionType.ALL
        )

        self.analysis_table = ddb.Table(
            self, "AnalysisTable",
            partition_key=ddb.Attribute(name="userid", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="date", type=ddb.AttributeType.STRING)
        )

        self.user_table = ddb.Table(
            self, "UserTable",
            partition_key=ddb.Attribute(name="userid", type=ddb.AttributeType.STRING)
        )

        nltk_layer = lambda_.LayerVersion(
            self, "NltkLayer",
            code=lambda_.Code.from_asset("layers/nltk"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description="NLTK"
        )

        # Define the Lambda layer as a variable
        pandas_layer = lambda_.LayerVersion(self, "PandasLayer",
            code=lambda_.Code.from_asset("layers/pandas"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            description="Pandas and Numpy"
        )

        # Define the Lambda functions
        self.preprocessing_lambda = self.create_lambda("PreprocessingLambda", "lib/preprocessing_lambda", environment={
                'RAW_BUCKET_NAME': self.raw_bucket.bucket_name,
                'PROCESSED_BUCKET_NAME': self.processed_bucket.bucket_name,
            })
        self.visualisation_lambda = self.create_lambda("VisualisationLambda", "lib/visualisation_lambda",layers=[pandas_layer], environment={
        'OUTPUT_BUCKET_NAME': self.output_bucket.bucket_name,
        'RESULTS_TABLE_NAME': self.results_table.table_name,
        'ANALYSIS_TABLE_NAME': self.analysis_table.table_name,
            })
        self.anomaly_lambda = self.create_lambda("AnomalyLambda", "lib/anomaly_lambda",layers=[pandas_layer], environment={
                'MALICIOUS_BUCKET_NAME': self.malicious_bucket.bucket_name,
                'PROCESSED_BUCKET_NAME': self.processed_bucket.bucket_name,
                'RESULTS_TABLE_NAME': self.results_table.table_name,
                'ANALYSIS_TABLE_NAME': self.analysis_table.table_name,
                'USER_TABLE_NAME': self.user_table.table_name,
                'VISUALISATION_LAMBDA_NAME': self.visualisation_lambda.function_name,
            })
        self.nltk_lambda = self.create_lambda("NltkLambda", "lib/nltk_lambda",layers=[nltk_layer], environment={
                'PROCESSED_BUCKET_NAME': self.processed_bucket.bucket_name,
                'RESULTS_TABLE_NAME': self.results_table.table_name,
                'ANOMALY_LAMBDA_NAME': self.anomaly_lambda.function_name,
            })
        self.comprehend_lambda = self.create_lambda("ComprehendLambda", "lib/comprehend_lambda", environment={
                'PROCESSED_BUCKET_NAME': self.processed_bucket.bucket_name,
                'RESULTS_TABLE_NAME': self.results_table.table_name,
                'USER_TABLE_NAME': self.user_table.table_name,
                'NLTK_LAMBDA_NAME': self.nltk_lambda.function_name,
            })




        # Granting permissions, because I am a single developer and not using predefined roles by a security team. Let AWS CDK manage the IAM roles

        self.malicious_bucket.grant_read(self.anomaly_lambda)

        self.raw_bucket.grant_read_write(self.preprocessing_lambda)
        self.raw_bucket.grant_read_write(self.visualisation_lambda)

        self.processed_bucket.grant_read_write(self.anomaly_lambda)
        self.processed_bucket.grant_read_write(self.comprehend_lambda)
        self.processed_bucket.grant_read_write(self.nltk_lambda)
        self.processed_bucket.grant_read_write(self.preprocessing_lambda)
        self.processed_bucket.grant_read_write(self.visualisation_lambda)

        self.output_bucket.grant_read_write(self.visualisation_lambda)

        self.results_table.grant_full_access(self.anomaly_lambda)
        self.results_table.grant_full_access(self.comprehend_lambda)
        self.results_table.grant_full_access(self.nltk_lambda)
        self.results_table.grant_full_access(self.visualisation_lambda)

        self.user_table.grant_full_access(self.anomaly_lambda)
        self.user_table.grant_full_access(self.comprehend_lambda)

        self.analysis_table.grant_full_access(self.anomaly_lambda)
        self.analysis_table.grant_full_access(self.visualisation_lambda)

        # Define the policy statement for Amazon Comprehend
        comprehend_policy_statement = iam.PolicyStatement(
            actions=[
                "comprehend:DetectDominantLanguage",
                "comprehend:DetectEntities",
                "comprehend:DetectSentiment",
                "comprehend:BatchDetectSentiment"
            ],
            resources=["*"],  # Comprehend does not support resource-level permissions
        )

        self.nltk_lambda.grant_invoke(self.comprehend_lambda)

        self.anomaly_lambda.grant_invoke(self.nltk_lambda)

        self.visualisation_lambda.grant_invoke(self.anomaly_lambda)

        # Add policy to comprehend lambda
        self.comprehend_lambda.add_to_role_policy(comprehend_policy_statement)

    def create_lambda(self, id, handler_directory,**kwargs):
        return lambda_.Function(
            self, id,
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="lambda_handler.lambda_handler",
            code=lambda_.Code.from_asset(handler_directory),
            **kwargs
        )
