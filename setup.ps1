# ============================================================
# 智慧大棚管理系统 - Conda 环境初始化脚本
# ============================================================
# 用法：
#   1. 在 PowerShell 中执行：  .\setup.ps1
#   2. 或在 cmd 中执行：       powershell -ExecutionPolicy Bypass -File setup.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$ENV_PREFIX   = Join-Path $PROJECT_ROOT ".conda-env"
$PKG_CACHE    = Join-Path $PROJECT_ROOT ".conda-cache"

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host " 智慧大棚管理系统 - 环境初始化" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "项目根目录: $PROJECT_ROOT"
Write-Host "Conda 环境: $ENV_PREFIX"

# 1. 配置 conda 包缓存到项目内（避免沙盒/权限问题）
$env:CONDA_PKGS_DIRS = $PKG_CACHE
$env:CONDA_NO_PLUGINS = "true"
if (-not (Test-Path $PKG_CACHE)) {
    New-Item -ItemType Directory -Force -Path $PKG_CACHE | Out-Null
}

# 2. 创建 conda 环境（克隆现有 langchain env 加速；若无则从 python 创建）
if (Test-Path (Join-Path $ENV_PREFIX "python.exe")) {
    Write-Host "[1/4] conda 环境已存在，跳过创建" -ForegroundColor Green
} else {
    $langchainEnv = conda env list | Select-String "langchain\s"
    if ($langchainEnv) {
        Write-Host "[1/4] 克隆 langchain 环境作为基础 ..." -ForegroundColor Yellow
        conda create --prefix $ENV_PREFIX --clone langchain -y
    } else {
        Write-Host "[1/4] 创建 Python 3.11 新环境 ..." -ForegroundColor Yellow
        conda create --prefix $ENV_PREFIX python=3.11 pip -y
    }
}

# 3. 安装/升级依赖
Write-Host "[2/4] 安装/升级依赖 ..." -ForegroundColor Yellow
$pythonExe = Join-Path $ENV_PREFIX "python.exe"
& $pythonExe -m pip install --upgrade `
    langgraph langchain langchain-core langchain-community `
    langchain-deepseek langchain-mcp-adapters mcp `
    openai streamlit pygwalker `
    pandas openpyxl python-dotenv requests

# 4. 复制 .env.example -> .env（若不存在）
$envFile = Join-Path $PROJECT_ROOT ".env"
$envExample = Join-Path $PROJECT_ROOT ".env.example"
if (-not (Test-Path $envFile)) {
    Write-Host "[3/4] 从 .env.example 创建 .env" -ForegroundColor Yellow
    Copy-Item $envExample $envFile
    Write-Host "      请编辑 .env 填入 DEEPSEEK_API_KEY" -ForegroundColor Red
} else {
    Write-Host "[3/4] .env 已存在" -ForegroundColor Green
}

# 5. 输出使用说明
Write-Host "[4/4] 完成" -ForegroundColor Green
Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host " 使用方式" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "激活环境："
Write-Host "  conda activate $ENV_PREFIX"
Write-Host ""
Write-Host "启动 Streamlit："
Write-Host "  conda activate $ENV_PREFIX"
Write-Host "  streamlit run app\main.py"
Write-Host ""
Write-Host "或直接用 python.exe："
Write-Host "  & `"$ENV_PREFIX\python.exe`" -m streamlit run app\main.py"
Write-Host ""
Write-Host "初始化病虫害数据库（可选）："
Write-Host "  & `"$ENV_PREFIX\python.exe`" scripts\init_pest_db.py"
Write-Host ""
Write-Host "启动 weather MCP 服务（可选）："
Write-Host "  & `"$ENV_PREFIX\python.exe`" -m weather_mcp.server"
Write-Host ""
Write-Host "别忘了在 .env 中填入 DEEPSEEK_API_KEY" -ForegroundColor Red
