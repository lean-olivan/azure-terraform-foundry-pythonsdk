# Terraform Infrastructure

This directory contains the Terraform configuration for deploying the Azure infrastructure.

## Files

- `main.tf` - Main infrastructure resources
- `variables.tf` - Input variables
- `terraform.tfvars` - Variable values
- `outputs.tf` - Output values
- `providers.tf` - Provider configuration

## Usage

### Initialize Terraform
```bash
cd terraform
terraform init
```

### Plan Infrastructure
```bash
terraform plan
```

### Apply Infrastructure
```bash
terraform apply
```

### Destroy Infrastructure
```bash
terraform destroy
```

### Validate Configuration
```bash
terraform validate
```

### Format Code
```bash
terraform fmt
```

## Resources Created

- Resource Group
- Azure OpenAI Account
- OpenAI Model Deployment
- Storage Account
- Blob Containers (uploads, results)
- Function App
- Service Plan
- Application Insights

## Notes

- Make sure you're logged into Azure before running `terraform apply`
- The function app uses a zip deployment from the `../function` directory
- All resources are tagged with the project and environment