variable "subscription_id" {
  description = "Azure Subscription ID"
  type        = string
}

variable "tenant_id" {
  description = "Azure Tenant ID"
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group to create"
  type        = string
  default     = "rg-testfoundry3-endpoint"
}

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "australiaeast"
}

variable "openai_account_name" {
  description = "Name of the Azure OpenAI account"
  type        = string
  default     = "testfoundry3-endpoint"
}

variable "model_deployment_name" {
  description = "Name of the model deployment"
  type        = string
  default     = "gpt-5.4-nano"
}

variable "model_name" {
  description = "OpenAI model name to deploy"
  type        = string
  default     = "gpt-5.4-nano"
}

variable "model_version" {
  description = "Version of the model to deploy"
  type        = string
  default     = "2026-03-17"
}

variable "model_capacity" {
  description = "Tokens per minute capacity (in thousands)"
  type        = number
  default     = 250
}

variable "storage_account_name" {
  description = "Name of the storage account for the function app"
  type        = string
  default     = "stfoundry3endpoint"
}

variable "function_app_name" {
  description = "Name of the Azure Function App"
  type        = string
  default     = "func-testfoundry3-endpoint"
}
