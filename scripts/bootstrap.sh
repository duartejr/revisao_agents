#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — Configuração inicial do Agente de Revisão da Literatura
# =============================================================================
# Uso:
#   chmod +x scripts/bootstrap.sh
#   ./scripts/bootstrap.sh
#
# Este script:
#   1. Verifica pré-requisitos (Python >= 3.11, uv, git)
#   2. Instala as dependências do projeto com `uv sync`
#   3. Gera o arquivo .env com suas chaves de API
#   4. Valida as variáveis obrigatórias
#   5. Exibe as instruções de início
# =============================================================================

set -euo pipefail

# ── Cores e utilitários ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # sem cor

ok()   { echo -e "${GREEN}  ✔ ${NC}$*"; }
warn() { echo -e "${YELLOW}  ⚠ ${NC}$*"; }
err()  { echo -e "${RED}  ✗ ${NC}$*"; }
info() { echo -e "${BLUE}  → ${NC}$*"; }
ask()  { echo -e "${CYAN}${BOLD}$*${NC}"; }

banner() {
  echo ""
  echo -e "${BOLD}${BLUE}============================================================${NC}"
  echo -e "${BOLD}${BLUE}  🔬 Agente de Revisão da Literatura — Bootstrap${NC}"
  echo -e "${BOLD}${BLUE}============================================================${NC}"
  echo ""
}

# ── Localização do script ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

banner
info "Diretório do projeto: $PROJECT_DIR"
echo ""

# =============================================================================
# FASE 1 — Verificação de pré-requisitos
# =============================================================================
echo -e "${BOLD}[1/4] Verificando pré-requisitos...${NC}"
echo ""

# Verificar git
if ! command -v git &>/dev/null; then
  err "git não encontrado. Instale o git antes de continuar."
  err "  Ubuntu/Debian: sudo apt install git"
  err "  Fedora:        sudo dnf install git"
  exit 1
fi
ok "git $(git --version | cut -d' ' -f3)"

# Verificar Python >= 3.11
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
    MAJOR=$(echo "$VER" | cut -d. -f1)
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [[ "$MAJOR" -gt 3 ]] || [[ "$MAJOR" -eq 3 && "$MINOR" -ge 11 ]]; then
      PYTHON_CMD="$cmd"
      break
    fi
  fi
done

if [[ -z "$PYTHON_CMD" ]]; then
  err "Python >= 3.11 não encontrado."
  err "  Instale Python 3.11+ em: https://www.python.org/downloads/"
  err "  Ou via pyenv: https://github.com/pyenv/pyenv"
  exit 1
fi
PYVER=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
ok "Python $PYVER"

# Verificar uv
if ! command -v uv &>/dev/null; then
  echo ""
  warn "uv não encontrado. Instalando automaticamente..."
  info "Mais informações: https://docs.astral.sh/uv/"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Adicionar ao PATH da sessão atual
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv &>/dev/null; then
    err "Falha ao instalar uv. Instale manualmente:"
    err "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    err "  Depois reinicie o terminal e execute este script novamente."
    exit 1
  fi
  ok "uv instalado com sucesso: $(uv --version)"
else
  ok "uv $(uv --version | cut -d' ' -f2)"
fi

echo ""

# =============================================================================
# FASE 2 — Instalação de dependências
# =============================================================================
echo -e "${BOLD}[2/4] Instalando dependências do projeto...${NC}"
echo ""

cd "$PROJECT_DIR"

info "Executando: uv sync"
if uv sync; then
  ok "Dependências instaladas com sucesso."
else
  err "Falha ao instalar dependências. Verifique o pyproject.toml e tente novamente."
  exit 1
fi

echo ""

# =============================================================================
# FASE 3 — Configuração do arquivo .env
# =============================================================================
echo -e "${BOLD}[3/4] Configurando credenciais (.env)...${NC}"
echo ""

ENV_FILE="$PROJECT_DIR/.env"
ENV_EXAMPLE="$PROJECT_DIR/.env.example"

if [[ ! -f "$ENV_EXAMPLE" ]]; then
  err "Arquivo .env.example não encontrado em $PROJECT_DIR"
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  warn "Arquivo .env já existe."
  ask "Deseja sobrescrever o .env existente? [s/N]"
  read -r OVERWRITE_ENV
  if [[ "$OVERWRITE_ENV" =~ ^[Ss]$ ]]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    ok "Arquivo .env recriado a partir de .env.example"
  else
    info "Mantendo .env existente. Pulando configuração de variáveis."
    SKIP_ENV_WIZARD=true
  fi
else
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  ok "Arquivo .env criado a partir de .env.example"
fi

if [[ "${SKIP_ENV_WIZARD:-false}" != "true" ]]; then
  echo ""
  echo -e "${BOLD}  Selecione o perfil de configuração:${NC}"
  echo "  [1] Mínimo  — OpenAI + Tavily + MongoDB (recomendado para começar)"
  echo "  [2] Completo — todos os provedores (OpenAI, Gemini, Groq, OpenRouter, Tavily, MongoDB)"
  echo ""
  ask "  Escolha [1/2, padrão=1]:"
  read -r PROFILE
  PROFILE="${PROFILE:-1}"

  echo ""
  set_env_var() {
    local KEY="$1"
    local PROMPT="$2"
    local DEFAULT="${3:-}"
    if [[ -n "$DEFAULT" ]]; then
      ask "  $PROMPT [padrão: $DEFAULT]:"
    else
      ask "  $PROMPT:"
    fi
    read -r VAL
    VAL="${VAL:-$DEFAULT}"
    if [[ -n "$VAL" ]]; then
      # Substitui ou adiciona a variável no .env
      if grep -q "^${KEY}=" "$ENV_FILE"; then
        sed -i "s|^${KEY}=.*|${KEY}=${VAL}|" "$ENV_FILE"
      else
        echo "${KEY}=${VAL}" >> "$ENV_FILE"
      fi
      ok "$KEY configurado."
    else
      warn "$KEY deixado em branco — será necessário preencher manualmente em .env"
    fi
  }

  # ── Provedor LLM principal ─────────────────────────────────────────────────
  echo -e "${BOLD}  ─── Provedor LLM principal ───────────────────────────────────${NC}"
  echo "  Opções: openai | google | groq | openrouter"
  ask "  LLM_PROVIDER [padrão: openai]:"
  read -r LLM_PROVIDER_VAL
  LLM_PROVIDER_VAL="${LLM_PROVIDER_VAL:-openai}"
  sed -i "s|^LLM_PROVIDER=.*|LLM_PROVIDER=${LLM_PROVIDER_VAL}|" "$ENV_FILE"
  ok "LLM_PROVIDER=${LLM_PROVIDER_VAL}"

  echo ""

  # ── Perfil mínimo ─────────────────────────────────────────────────────────
  echo -e "${BOLD}  ─── OpenAI (obrigatório no perfil mínimo) ─────────────────────${NC}"
  info "  Obtenha sua chave em: https://platform.openai.com/api-keys"
  set_env_var "OPENAI_API_KEY" "OPENAI_API_KEY (começa com sk-...)"

  echo ""
  echo -e "${BOLD}  ─── Tavily Search (obrigatório no perfil mínimo) ──────────────${NC}"
  info "  Obtenha sua chave em: https://app.tavily.com"
  set_env_var "TAVILY_API_KEY" "TAVILY_API_KEY (começa com tvly-...)"

  echo ""
  echo -e "${BOLD}  ─── MongoDB (obrigatório no perfil mínimo) ─────────────────────${NC}"
  info "  Atlas gratuito em: https://www.mongodb.com/atlas"
  info "  Local padrão: mongodb://localhost:27017"
  set_env_var "MONGODB_URI" "MONGODB_URI" "mongodb://localhost:27017"
  set_env_var "MONGODB_DB"  "MONGODB_DB (nome do banco)" "revisao_agents"

  # ── Perfil completo ───────────────────────────────────────────────────────
  if [[ "$PROFILE" == "2" ]]; then
    echo ""
    echo -e "${BOLD}  ─── Perfil completo: provedores adicionais ─────────────────────${NC}"

    echo ""
    echo -e "  ${CYAN}Google Gemini:${NC}"
    info "  Obtenha sua chave em: https://aistudio.google.com/apikey"
    set_env_var "GOOGLE_API_KEY" "GOOGLE_API_KEY"

    echo ""
    echo -e "  ${CYAN}Groq:${NC}"
    info "  Obtenha sua chave em: https://console.groq.com/keys"
    set_env_var "GROQ_API_KEY" "GROQ_API_KEY (começa com gsk_...)"

    echo ""
    echo -e "  ${CYAN}OpenRouter:${NC}"
    info "  Obtenha sua chave em: https://openrouter.ai/keys"
    set_env_var "OPENROUTER_API_KEY" "OPENROUTER_API_KEY (começa com sk-or-...)"
  fi
fi

echo ""

# =============================================================================
# FASE 4 — Validação das variáveis obrigatórias
# =============================================================================
echo -e "${BOLD}[4/4] Validando configuração...${NC}"
echo ""

# Carrega as variáveis do .env para verificação (sem exportar para o shell)
check_var() {
  local KEY="$1"
  local VAL
  VAL=$(grep "^${KEY}=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2-)
  if [[ -z "$VAL" || "$VAL" == *"..."* || "$VAL" == "sk-..." || "$VAL" == "tvly-..." ]]; then
    warn "$KEY não configurado ou ainda com valor de exemplo."
    return 1
  else
    ok "$KEY ✔"
    return 0
  fi
}

MISSING=0
check_var "OPENAI_API_KEY" || MISSING=$((MISSING+1))
check_var "TAVILY_API_KEY" || MISSING=$((MISSING+1))
check_var "MONGODB_URI"    || MISSING=$((MISSING+1))

echo ""

# =============================================================================
# Resultado final
# =============================================================================
echo -e "${BOLD}${BLUE}============================================================${NC}"
echo -e "${BOLD}${BLUE}  🎉 Bootstrap concluído!${NC}"
echo -e "${BOLD}${BLUE}============================================================${NC}"
echo ""

if [[ "$MISSING" -gt 0 ]]; then
  warn "$MISSING variável(eis) obrigatória(s) não configurada(s)."
  warn "Edite o arquivo .env antes de iniciar:"
  warn "  nano .env   (ou qualquer editor de sua preferência)"
  echo ""
fi

echo -e "${BOLD}  Para iniciar a interface gráfica (UI):${NC}"
echo -e "  ${GREEN}uv run python run_ui.py${NC}"
echo ""
echo -e "${BOLD}  Para iniciar pelo menu interativo (CLI):${NC}"
echo -e "  ${GREEN}uv run python -m revisao_agents${NC}"
echo ""
echo -e "${BOLD}  Para usar a CLI script diretamente:${NC}"
echo -e "  ${GREEN}uv run revisao-agents --help${NC}"
echo ""
echo -e "  Acesse a UI em: ${CYAN}http://localhost:7860${NC}"
echo ""
echo -e "  Documentação completa: ${CYAN}docs/README.md${NC}"
echo ""
