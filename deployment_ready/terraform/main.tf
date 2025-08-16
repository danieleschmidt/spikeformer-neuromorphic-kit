# Spikeformer Global Infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Multi-region deployment
module "spikeformer_us_east_1" {
  source = "./modules/spikeformer"
  
  region                = "us-east-1"
  environment          = var.environment
  instance_type        = var.instance_type
  min_capacity         = var.min_capacity
  max_capacity         = var.max_capacity
  
  tags = local.common_tags
}

module "spikeformer_eu_west_1" {
  source = "./modules/spikeformer"
  
  region                = "eu-west-1"
  environment          = var.environment
  instance_type        = var.instance_type
  min_capacity         = var.min_capacity
  max_capacity         = var.max_capacity
  
  tags = local.common_tags
}

module "spikeformer_ap_southeast_1" {
  source = "./modules/spikeformer"
  
  region                = "ap-southeast-1"
  environment          = var.environment
  instance_type        = var.instance_type
  min_capacity         = var.min_capacity
  max_capacity         = var.max_capacity
  
  tags = local.common_tags
}

# Global load balancer
resource "aws_route53_zone" "spikeformer" {
  name = "spikeformer.ai"
  
  tags = local.common_tags
}

locals {
  common_tags = {
    Project     = "Spikeformer"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
