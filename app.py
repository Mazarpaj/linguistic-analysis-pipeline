#!/usr/bin/env python3

from aws_cdk import App

from pipeline_stack import PipelineStack


app = App()

PipelineStack(app, "PipelineStack", env={'region': 'eu-west-2'})

app.synth() 