import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { aws_ecs as ecs } from 'aws-cdk-lib';
import { aws_ecs_patterns as ecs_patterns } from 'aws-cdk-lib';
import { aws_route53 as route53 } from 'aws-cdk-lib';
import { aws_certificatemanager as acm } from 'aws-cdk-lib';
import { aws_route53_targets as targets } from 'aws-cdk-lib';
import { aws_ec2 as ec2 } from 'aws-cdk-lib';

export class TiptoeStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const DOMAIN_NAME = 'exolabs.net';
    const ENCODING_DOMAIN = `search-enc.${DOMAIN_NAME}`;
    const EMBEDDING_DOMAIN = `search-emb.${DOMAIN_NAME}`;

    // Look up the existing hosted zone
    const hostedZone = route53.HostedZone.fromLookup(this, 'HostedZone', {
      domainName: DOMAIN_NAME
    });

    // Use existing certificate
    const certificateArn = 'arn:aws:acm:us-east-1:559050203951:certificate/d5f4d75b-5d0c-4716-a42d-e731be0815bc';
    const certificate = acm.Certificate.fromCertificateArn(this, 'ExistingCertificate', certificateArn);

    // Create VPC
    const vpc = new ec2.Vpc(this, 'TiptoeVPC', {
      maxAzs: 2,
      natGateways: 1,
    });

    // Create ECS cluster
    const cluster = new ecs.Cluster(this, 'TiptoeCluster', {
      vpc,
      clusterName: 'tiptoe-cluster',
      containerInsights: true,
    });

    // Create Fargate service for encoding server
    const encodingService = new ecs_patterns.ApplicationLoadBalancedFargateService(this, 'EncodingService', {
      cluster,
      serviceName: 'tiptoe-encoding-server',
      taskImageOptions: {
        image: ecs.ContainerImage.fromAsset('..', {
          file: 'tiptoe/Dockerfile',
          exclude: [
            'target', 
            '**/target',
            'cdk.out',
            '**/cdk.out',
            'node_modules',
            '**/node_modules'
          ],
        }),
        containerName: 'web',
        command: ['./encoding_server'],
        environment: {
          CARGO_MANIFEST_DIR: '/app',
          VIRTUAL_ENV: '/opt/venv',
          PATH: '/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        },
        containerPort: 3000,
      },
      cpu: 1024,
      memoryLimitMiB: 2048,
      desiredCount: 1,
      publicLoadBalancer: true,
      certificate: certificate,
      domainName: ENCODING_DOMAIN,
      domainZone: hostedZone,
      redirectHTTP: true,
      assignPublicIp: true,
    });

    // Create Fargate service for embedding server
    const embeddingService = new ecs_patterns.ApplicationLoadBalancedFargateService(this, 'EmbeddingService', {
      cluster,
      serviceName: 'tiptoe-embedding-server',
      taskImageOptions: {
        image: ecs.ContainerImage.fromAsset('..', {
          file: 'tiptoe/Dockerfile',
          exclude: [
            'target', 
            '**/target',
            'cdk.out',
            '**/cdk.out',
            'node_modules',
            '**/node_modules'
          ],
        }),
        containerName: 'web',
        command: ['./embedding_server'],
        environment: {
          CARGO_MANIFEST_DIR: '/app',
          VIRTUAL_ENV: '/opt/venv',
          PATH: '/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        },
        containerPort: 3001,
      },
      cpu: 1024,
      memoryLimitMiB: 2048,
      desiredCount: 1,
      publicLoadBalancer: true,
      certificate: certificate,
      domainName: EMBEDDING_DOMAIN,
      domainZone: hostedZone,
      redirectHTTP: true,
      assignPublicIp: true,
    });

    // Configure health checks with more lenient settings
    encodingService.targetGroup.configureHealthCheck({
      path: '/params',
      interval: cdk.Duration.seconds(60),
      timeout: cdk.Duration.seconds(30),
      healthyThresholdCount: 2,
      unhealthyThresholdCount: 5,
    });

    embeddingService.targetGroup.configureHealthCheck({
      path: '/params',
      interval: cdk.Duration.seconds(60),
      timeout: cdk.Duration.seconds(30),
      healthyThresholdCount: 2,
      unhealthyThresholdCount: 5,
    });

    // Output the domain names
    new cdk.CfnOutput(this, 'EncodingServerDomain', {
      value: `https://${ENCODING_DOMAIN}`,
      description: 'Encoding Server Domain',
    });

    new cdk.CfnOutput(this, 'EmbeddingServerDomain', {
      value: `https://${EMBEDDING_DOMAIN}`,
      description: 'Embedding Server Domain',
    });
  }
}