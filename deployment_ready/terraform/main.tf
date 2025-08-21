# SpikeFormer Multi-Region Deployment Configuration

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "spikeformer-terraform-state"
    key    = "global/terraform.tfstate"
    region = "us-east-1"
  }
}

# Provider configurations for each region
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

provider "aws" {
  alias  = "us_west_2"
  region = "us-west-2"
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
}

provider "aws" {
  alias  = "eu_central_1"
  region = "eu-central-1"
}

provider "aws" {
  alias  = "ap_southeast_1"
  region = "ap-southeast-1"
}

provider "aws" {
  alias  = "ap_northeast_1"
  region = "ap-northeast-1"
}

provider "aws" {
  alias  = "ap_south_1"
  region = "ap-south-1"
}

# Global resources
resource "aws_route53_zone" "main" {
  name = var.domain_name
  
  tags = {
    Name        = "SpikeFormer Main Zone"
    Environment = var.environment
    Project     = "SpikeFormer"
  }
}

# Global WAF
resource "aws_wafv2_web_acl" "global" {
  name  = "spikeformer-global-waf"
  scope = "CLOUDFRONT"

  default_action {
    allow {}
  }

  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSetMetric"
      sampled_requests_enabled   = true
    }
  }

  tags = {
    Name        = "SpikeFormer Global WAF"
    Environment = var.environment
  }
}

# Regional deployments
module "us_east_1" {
  source = "./modules/regional-deployment"
  
  providers = {
    aws = aws.us_east_1
  }
  
  region             = "us-east-1"
  environment        = var.environment
  instance_type      = var.instance_types["us-east-1"]
  min_capacity       = var.min_capacities["us-east-1"]
  max_capacity       = var.max_capacities["us-east-1"]
  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

module "us_west_2" {
  source = "./modules/regional-deployment"
  
  providers = {
    aws = aws.us_west_2
  }
  
  region             = "us-west-2"
  environment        = var.environment
  instance_type      = var.instance_types["us-west-2"]
  min_capacity       = var.min_capacities["us-west-2"]
  max_capacity       = var.max_capacities["us-west-2"]
  vpc_cidr           = "10.1.0.0/16"
  availability_zones = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

module "eu_west_1" {
  source = "./modules/regional-deployment"
  
  providers = {
    aws = aws.eu_west_1
  }
  
  region             = "eu-west-1"
  environment        = var.environment
  instance_type      = var.instance_types["eu-west-1"]
  min_capacity       = var.min_capacities["eu-west-1"]
  max_capacity       = var.max_capacities["eu-west-1"]
  vpc_cidr           = "10.2.0.0/16"
  availability_zones = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]
}

# CloudFront distribution for global CDN
resource "aws_cloudfront_distribution" "global" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "SpikeFormer Global Distribution"
  default_root_object = "index.html"
  web_acl_id          = aws_wafv2_web_acl.global.arn

  origin {
    domain_name = module.us_east_1.load_balancer_dns_name
    origin_id   = "primary-origin"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "primary-origin"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "CloudFront-Forwarded-Proto"]

      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = var.ssl_certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name        = "SpikeFormer Global CDN"
    Environment = var.environment
  }
}

# Route53 health checks for each region
resource "aws_route53_health_check" "regions" {
  for_each = toset(var.regions)

  fqdn                            = "${each.key}.api.${var.domain_name}"
  port                            = 443
  type                            = "HTTPS"
  resource_path                   = "/health"
  failure_threshold               = "3"
  request_interval                = "30"
  cloudwatch_alarm_region         = each.key
  cloudwatch_alarm_name           = "spikeformer-${each.key}-health"
  insufficient_data_health_status = "Failure"

  tags = {
    Name   = "SpikeFormer ${each.key} Health Check"
    Region = each.key
  }
}

# Global monitoring and alerting
resource "aws_cloudwatch_metric_alarm" "global_error_rate" {
  alarm_name          = "spikeformer-global-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "4XXError"
  namespace           = "AWS/CloudFront"
  period              = "300"
  statistic           = "Sum"
  threshold           = "50"
  alarm_description   = "This metric monitors global error rate"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    DistributionId = aws_cloudfront_distribution.global.id
  }

  tags = {
    Name        = "SpikeFormer Global Error Rate"
    Environment = var.environment
  }
}

# SNS topic for global alerts
resource "aws_sns_topic" "alerts" {
  name = "spikeformer-global-alerts"

  tags = {
    Name        = "SpikeFormer Global Alerts"
    Environment = var.environment
  }
}

# Output values
output "cloudfront_distribution_domain" {
  value = aws_cloudfront_distribution.global.domain_name
}

output "route53_zone_id" {
  value = aws_route53_zone.main.zone_id
}

output "regional_endpoints" {
  value = {
    us_east_1      = module.us_east_1.load_balancer_dns_name
    us_west_2      = module.us_west_2.load_balancer_dns_name
    eu_west_1      = module.eu_west_1.load_balancer_dns_name
  }
}
