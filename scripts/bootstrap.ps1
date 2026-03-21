# =============================================================================
# bootstrap.ps1 — Configuração inicial do Agente de Revisão da Literatura
# =============================================================================
# Uso (PowerShell como Administrador ou com política de execução liberada):
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#   .\scripts\bootstrap.ps1
#
# Este script:
#   1. Verifica pré-requisitos (Python >= 3.11, uv, git)
#   2. Instala as dependências do projeto com `uv sync`
#   3. Gera o arquivo .env com suas chaves de API
#   4. Valida as variáveis obrigatórias
#   5. Exibe as instruções de início
# =============================================================================

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

# ── Utilitários ───────────────────────────────────────────────────────────────
function Write-Ok   { param($msg) Write-Host "  $([char]0x2714) $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  $([char]0x26A0) $msg" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "  $([char]0x2717) $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "  -> $msg" -ForegroundColor Cyan }
function Write-Ask  { param($msg) Write-Host $msg -ForegroundColor Cyan -NoNewline }

function Show-Banner {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Blue
    Write-Host "  Agente de Revisao da Literatura -- Bootstrap" -ForegroundColor Blue
    Write-Host "============================================================" -ForegroundColor Blue
    Write-Host ""
}

function Set-EnvVar {
    param(
        [string]$Key,
        [string]$Prompt,
        [string]$Default = ""
    )
    if ($Default) {
        Write-Ask "  $Prompt [padrao: $Default]: "
    } else {
        Write-Ask "  ${Prompt}: "
    }
    $val = Read-Host
    if ([string]::IsNullOrWhiteSpace($val)) { $val = $Default }
    if (-not [string]::IsNullOrWhiteSpace($val)) {
        $content = Get-Content $envFile -Raw
        if ($content -match "(?m)^${Key}=.*$") {
            $content = $content -replace "(?m)^${Key}=.*$", "${Key}=${val}"
        } else {
            $content += "`n${Key}=${val}"
        }
        Set-Content $envFile -Value $content -NoNewline
        Write-Ok "$Key configurado."
    } else {
        Write-Warn "$Key deixado em branco -- sera necessario preencher manualmente no .env"
    }
}

# ── Localização do projeto ────────────────────────────────────────────────────
$scriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir

Show-Banner
Write-Info "Diretorio do projeto: $projectDir"
Write-Host ""

# =============================================================================
# FASE 1 — Verificação de pré-requisitos
# =============================================================================
Write-Host "[1/4] Verificando pre-requisitos..." -ForegroundColor White
Write-Host ""

# Verificar git
try {
    $gitVer = (git --version 2>&1).ToString().Trim()
    Write-Ok $gitVer
} catch {
    Write-Err "git nao encontrado. Instale o git antes de continuar."
    Write-Err "  Download: https://git-scm.com/download/win"
    exit 1
}

# Verificar Python >= 3.11
$pythonCmd = $null
foreach ($cmd in @("python3.12", "python3.11", "python3", "python")) {
    try {
        $verStr = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($verStr) {
            $parts = $verStr.Split(".")
            $major = [int]$parts[0]
            $minor = [int]$parts[1]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 11)) {
                $pythonCmd = $cmd
                break
            }
        }
    } catch { }
}

if (-not $pythonCmd) {
    Write-Err "Python >= 3.11 nao encontrado."
    Write-Err "  Instale Python 3.11+ em: https://www.python.org/downloads/"
    exit 1
}
$pyVer = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-Ok "Python $pyVer"

# Verificar uv
$uvAvailable = $false
try {
    $uvVer = (uv --version 2>&1).ToString().Trim()
    Write-Ok "uv $($uvVer -replace 'uv ','')"
    $uvAvailable = $true
} catch { }

if (-not $uvAvailable) {
    Write-Host ""
    Write-Warn "uv nao encontrado. Instalando automaticamente..."
    Write-Info "Mais informacoes: https://docs.astral.sh/uv/"
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
        $uvVer = (uv --version 2>&1).ToString().Trim()
        Write-Ok "uv instalado com sucesso: $uvVer"
    } catch {
        Write-Err "Falha ao instalar uv automaticamente."
        Write-Err "  Instale manualmente: https://docs.astral.sh/uv/getting-started/installation/"
        Write-Err "  Depois reinicie o PowerShell e execute este script novamente."
        exit 1
    }
}

Write-Host ""

# =============================================================================
# FASE 2 — Instalação de dependências
# =============================================================================
Write-Host "[2/4] Instalando dependencias do projeto..." -ForegroundColor White
Write-Host ""

Set-Location $projectDir

Write-Info "Executando: uv sync"
try {
    uv sync
    Write-Ok "Dependencias instaladas com sucesso."
} catch {
    Write-Err "Falha ao instalar dependencias. Verifique o pyproject.toml e tente novamente."
    exit 1
}

Write-Host ""

# =============================================================================
# FASE 3 — Configuração do arquivo .env
# =============================================================================
Write-Host "[3/4] Configurando credenciais (.env)..." -ForegroundColor White
Write-Host ""

$envFile    = Join-Path $projectDir ".env"
$envExample = Join-Path $projectDir ".env.example"

if (-not (Test-Path $envExample)) {
    Write-Err "Arquivo .env.example nao encontrado em $projectDir"
    exit 1
}

$skipEnvWizard = $false
if (Test-Path $envFile) {
    Write-Warn "Arquivo .env ja existe."
    Write-Ask "Deseja sobrescrever o .env existente? [s/N]: "
    $overwrite = Read-Host
    if ($overwrite -match "^[Ss]$") {
        Copy-Item $envExample $envFile -Force
        Write-Ok "Arquivo .env recriado a partir de .env.example"
    } else {
        Write-Info "Mantendo .env existente. Pulando configuracao de variaveis."
        $skipEnvWizard = $true
    }
} else {
    Copy-Item $envExample $envFile
    Write-Ok "Arquivo .env criado a partir de .env.example"
}

if (-not $skipEnvWizard) {
    Write-Host ""
    Write-Host "  Selecione o perfil de configuracao:" -ForegroundColor White
    Write-Host "  [1] Minimo  -- OpenAI + Tavily + MongoDB (recomendado para comecar)"
    Write-Host "  [2] Completo -- todos os provedores (OpenAI, Gemini, Groq, OpenRouter, Tavily, MongoDB)"
    Write-Host ""
    Write-Ask "  Escolha [1/2, padrao=1]: "
    $profile = Read-Host
    if ([string]::IsNullOrWhiteSpace($profile)) { $profile = "1" }

    Write-Host ""

    # ── Provedor LLM principal ─────────────────────────────────────────────
    Write-Host "  --- Provedor LLM principal ---" -ForegroundColor White
    Write-Host "  Opcoes: openai | google | groq | openrouter"
    Write-Ask "  LLM_PROVIDER [padrao: openai]: "
    $llmProvider = Read-Host
    if ([string]::IsNullOrWhiteSpace($llmProvider)) { $llmProvider = "openai" }
    $content = Get-Content $envFile -Raw
    $content = $content -replace "(?m)^LLM_PROVIDER=.*$", "LLM_PROVIDER=$llmProvider"
    Set-Content $envFile -Value $content -NoNewline
    Write-Ok "LLM_PROVIDER=$llmProvider"

    Write-Host ""

    # ── OpenAI ───────────────────────────────────────────────────────────────
    Write-Host "  --- OpenAI (obrigatorio no perfil minimo) ---" -ForegroundColor White
    Write-Info "  Obtenha sua chave em: https://platform.openai.com/api-keys"
    Set-EnvVar -Key "OPENAI_API_KEY" -Prompt "OPENAI_API_KEY (comeca com sk-...)"

    Write-Host ""
    # ── Tavily ────────────────────────────────────────────────────────────────
    Write-Host "  --- Tavily Search (obrigatorio no perfil minimo) ---" -ForegroundColor White
    Write-Info "  Obtenha sua chave em: https://app.tavily.com"
    Set-EnvVar -Key "TAVILY_API_KEY" -Prompt "TAVILY_API_KEY (comeca com tvly-...)"

    Write-Host ""
    # ── MongoDB ───────────────────────────────────────────────────────────────
    Write-Host "  --- MongoDB (obrigatorio no perfil minimo) ---" -ForegroundColor White
    Write-Info "  Atlas gratuito em: https://www.mongodb.com/atlas"
    Write-Info "  Local padrao: mongodb://localhost:27017"
    Set-EnvVar -Key "MONGODB_URI" -Prompt "MONGODB_URI" -Default "mongodb://localhost:27017"
    Set-EnvVar -Key "MONGODB_DB"  -Prompt "MONGODB_DB (nome do banco)" -Default "revisao_agents"

    # ── Perfil completo ───────────────────────────────────────────────────────
    if ($profile -eq "2") {
        Write-Host ""
        Write-Host "  --- Perfil completo: provedores adicionais ---" -ForegroundColor White

        Write-Host ""
        Write-Host "  Google Gemini:" -ForegroundColor Cyan
        Write-Info "  Obtenha sua chave em: https://aistudio.google.com/apikey"
        Set-EnvVar -Key "GOOGLE_API_KEY" -Prompt "GOOGLE_API_KEY"

        Write-Host ""
        Write-Host "  Groq:" -ForegroundColor Cyan
        Write-Info "  Obtenha sua chave em: https://console.groq.com/keys"
        Set-EnvVar -Key "GROQ_API_KEY" -Prompt "GROQ_API_KEY (comeca com gsk_...)"

        Write-Host ""
        Write-Host "  OpenRouter:" -ForegroundColor Cyan
        Write-Info "  Obtenha sua chave em: https://openrouter.ai/keys"
        Set-EnvVar -Key "OPENROUTER_API_KEY" -Prompt "OPENROUTER_API_KEY (comeca com sk-or-...)"
    }
}

Write-Host ""

# =============================================================================
# FASE 4 — Validação das variáveis obrigatórias
# =============================================================================
Write-Host "[4/4] Validando configuracao..." -ForegroundColor White
Write-Host ""

function Check-EnvVar {
    param([string]$Key)
    $content = Get-Content $envFile -Raw
    if ($content -match "(?m)^${Key}=(.+)$") {
        $val = $Matches[1].Trim()
        if ([string]::IsNullOrWhiteSpace($val) -or $val -match "\.\.\." -or $val -eq "sk-..." -or $val -eq "tvly-...") {
            Write-Warn "$Key nao configurado ou ainda com valor de exemplo."
            return $false
        } else {
            Write-Ok "$Key OK"
            return $true
        }
    } else {
        Write-Warn "$Key nao encontrado no .env"
        return $false
    }
}

$missing = 0
if (-not (Check-EnvVar "OPENAI_API_KEY")) { $missing++ }
if (-not (Check-EnvVar "TAVILY_API_KEY")) { $missing++ }
if (-not (Check-EnvVar "MONGODB_URI"))    { $missing++ }

Write-Host ""

# =============================================================================
# Resultado final
# =============================================================================
Write-Host "============================================================" -ForegroundColor Blue
Write-Host "  Agente de Revisao da Literatura -- Bootstrap concluido!" -ForegroundColor Blue
Write-Host "============================================================" -ForegroundColor Blue
Write-Host ""

if ($missing -gt 0) {
    Write-Warn "$missing variavel(eis) obrigatoria(s) nao configurada(s)."
    Write-Warn "Edite o arquivo .env antes de iniciar:"
    Write-Warn "  notepad .env"
    Write-Host ""
}

Write-Host "  Para iniciar a interface grafica (UI):" -ForegroundColor White
Write-Host "  uv run python run_ui.py" -ForegroundColor Green
Write-Host ""
Write-Host "  Para iniciar pelo menu interativo (CLI):" -ForegroundColor White
Write-Host "  uv run python -m revisao_agents" -ForegroundColor Green
Write-Host ""
Write-Host "  Para usar a CLI script diretamente:" -ForegroundColor White
Write-Host "  uv run revisao-agents --help" -ForegroundColor Green
Write-Host ""
Write-Host "  Acesse a UI em: http://localhost:7860" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Documentacao completa: docs\README.md" -ForegroundColor Cyan
Write-Host ""
