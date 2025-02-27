#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { TiptoeStack } from '../lib/tiptoe-cdk-stack';

const app = new cdk.App();
new TiptoeStack(app, 'TiptoeStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});