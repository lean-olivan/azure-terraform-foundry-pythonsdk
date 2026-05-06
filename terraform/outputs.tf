output "resource_group_name" {
  description = "Name of the created resource group"
  value       = azurerm_resource_group.rg.name
}

output "openai_account_name" {
  description = "Name of the Azure OpenAI account"
  value       = azurerm_cognitive_account.openai.name
}

output "openai_endpoint" {
  description = "Azure OpenAI base endpoint URL"
  value       = azurerm_cognitive_account.openai.endpoint
}

output "openai_responses_endpoint" {
  description = "Azure OpenAI Responses API endpoint (ready to use)"
  value       = "${azurerm_cognitive_account.openai.endpoint}openai/responses?api-version=2025-04-01-preview"
}

output "openai_chat_completions_endpoint" {
  description = "Azure OpenAI Chat Completions endpoint for the deployed model"
  value       = "${azurerm_cognitive_account.openai.endpoint}openai/deployments/${azurerm_cognitive_deployment.gpt_5_4_nano.name}/chat/completions?api-version=2025-04-01-preview"
}

output "openai_resource_id" {
  description = "Resource ID of the Azure OpenAI account"
  value       = azurerm_cognitive_account.openai.id
}

output "model_deployment_name" {
  description = "Name of the deployed model"
  value       = azurerm_cognitive_deployment.gpt_5_4_nano.name
}

output "openai_primary_key" {
  description = "Primary access key for the Azure OpenAI account"
  value       = azurerm_cognitive_account.openai.primary_access_key
  sensitive   = true
}

output "function_app_name" {
  description = "Name of the Azure Function App"
  value       = azurerm_linux_function_app.function.name
}

output "function_app_url" {
  description = "URL of the Azure Function App"
  value       = azurerm_linux_function_app.function.default_hostname
}

output "function_ask_endpoint" {
  description = "Full URL for the ask function endpoint"
  value       = "https://${azurerm_linux_function_app.function.default_hostname}/api/ask"
}
