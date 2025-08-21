# SpikeFormer Terraform Variables

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "domain_name" {
  description = "Primary domain name"
  type        = string
  default     = "spikeformer.ai"
}

variable "ssl_certificate_arn" {
  description = "SSL certificate ARN for CloudFront"
  type        = string
}

variable "regions" {
  description = "List of AWS regions to deploy to"
  type        = list(string)
  default     = ["us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1", "ap-south-1"]
}

variable "instance_types" {
  description = "EC2 instance types per region"
  type        = map(string)
  default = {
    us-east-1      = "c5.xlarge"
    us-west-2      = "c5.large"
    eu-west-1      = "c5.large"
    eu-central-1   = "c5.large"
    ap-southeast-1 = "c5.large"
    ap-northeast-1 = "c5.large"
    ap-south-1     = "c5.large"
  }
}

variable "min_capacities" {
  description = "Minimum capacity per region"
  type        = map(number)
  default = {
    us-east-1      = 2
    us-west-2      = 1
    eu-west-1      = 1
    eu-central-1   = 1
    ap-southeast-1 = 1
    ap-northeast-1 = 1
    ap-south-1     = 1
  }
}

variable "max_capacities" {
  description = "Maximum capacity per region"
  type        = map(number)
  default = {
    us-east-1      = 20
    us-west-2      = 10
    eu-west-1      = 15
    eu-central-1   = 10
    ap-southeast-1 = 12
    ap-northeast-1 = 15
    ap-south-1     = 8
  }
}

variable "monitoring_retention_days" {
  description = "CloudWatch logs retention in days"
  type        = number
  default     = 30
}

variable "backup_retention_days" {
  description = "Backup retention in days"
  type        = number
  default     = 30
}

# Neuromorphic-specific variables
variable "consciousness_threshold" {
  description = "Consciousness detection threshold"
  type        = number
  default     = 0.85
}

variable "quantum_coherence_target" {
  description = "Target quantum coherence level"
  type        = number
  default     = 0.95
}

variable "transcendence_enabled" {
  description = "Enable transcendence features"
  type        = bool
  default     = true
}

variable "multiverse_optimization" {
  description = "Enable multiverse optimization"
  type        = bool
  default     = true
}
