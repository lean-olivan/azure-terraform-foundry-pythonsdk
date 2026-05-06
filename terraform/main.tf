# Resource Group
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location

  tags = {
    project     = "testfoundry3-endpoint"
    environment = "dev"
    managed_by  = "terraform"
  }
}

# Create zip file for function deployment
data "archive_file" "function_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../function"
  output_path = "${path.module}/function/function.zip"
  excludes = concat(
    [".venv"],
    ["__pycache__"],
    [for f in fileset("${path.module}/../function/.venv", "**") : ".venv/${f}"],
    [for f in fileset("${path.module}/../function/**/__pycache__", "**") : "**/__pycache__/${f}"]
  )
}

# Azure OpenAI Account
resource "azurerm_cognitive_account" "openai" {
  name                = var.openai_account_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  kind                = "OpenAI"
  sku_name            = "S0"

  tags = {
    project     = "testfoundry3-endpoint"
    environment = "dev"
    managed_by  = "terraform"
  }
}

# Model Deployment: gpt-5.4-nano
resource "azurerm_cognitive_deployment" "gpt_5_4_nano" {
  name                 = var.model_deployment_name
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = var.model_name
    version = var.model_version
  }

  sku {
    name     = "GlobalStandard"
    capacity = var.model_capacity
  }
}

# Storage Account for Function App
resource "azurerm_storage_account" "function" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  tags = {
    project     = "testfoundry3-endpoint"
    environment = "dev"
    managed_by  = "terraform"
  }
}

# Blob container for incoming .txt uploads (Path C)
resource "azurerm_storage_container" "uploads" {
  name                  = "uploads"
  storage_account_id    = azurerm_storage_account.function.id
  container_access_type = "private"
}

# Blob container for processed results (Path C)
resource "azurerm_storage_container" "results" {
  name                  = "results"
  storage_account_id    = azurerm_storage_account.function.id
  container_access_type = "private"
}

# Service Plan for Function App
resource "azurerm_service_plan" "function" {
  name                = "plan-${var.function_app_name}"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  os_type             = "Linux"
  sku_name            = "Y1" # Consumption plan

  tags = {
    project     = "testfoundry3-endpoint"
    environment = "dev"
    managed_by  = "terraform"
  }
}

# Application Insights for Function diagnostics
resource "azurerm_application_insights" "function" {
  name                = "appi-${var.function_app_name}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"

  tags = {
    project     = "testfoundry3-endpoint"
    environment = "dev"
    managed_by  = "terraform"
  }
}

# Function App
resource "azurerm_linux_function_app" "function" {
  name                = var.function_app_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location

  storage_account_name       = azurerm_storage_account.function.name
  storage_account_access_key = azurerm_storage_account.function.primary_access_key
  service_plan_id            = azurerm_service_plan.function.id

  site_config {
    application_insights_connection_string = azurerm_application_insights.function.connection_string
    application_insights_key               = azurerm_application_insights.function.instrumentation_key

    application_stack {
      python_version = "3.11"
    }
  }

  app_settings = {
    "FUNCTIONS_WORKER_RUNTIME" = "python"
    "AzureWebJobsStorage"      = "DefaultEndpointsProtocol=https;AccountName=${azurerm_storage_account.function.name};AccountKey=${azurerm_storage_account.function.primary_access_key};EndpointSuffix=core.windows.net"
    "OPENAI_ENDPOINT"          = azurerm_cognitive_account.openai.endpoint
    "OPENAI_API_KEY"           = azurerm_cognitive_account.openai.primary_access_key
    "OPENAI_MODEL"             = var.model_deployment_name
    "STORAGE_ACCOUNT_NAME"     = azurerm_storage_account.function.name
    "STORAGE_ACCOUNT_KEY"      = azurerm_storage_account.function.primary_access_key
  }

  timeouts {
    create = "30m"
    update = "30m"
    delete = "30m"
  }

  tags = {
    project     = "testfoundry3-endpoint"
    environment = "dev"
    managed_by  = "terraform"
  }

  zip_deploy_file = data.archive_file.function_zip.output_path
}