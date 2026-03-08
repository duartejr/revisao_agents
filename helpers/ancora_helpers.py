"""
ancora_helpers.py
=================
Funções auxiliares para trabalhar com âncoras e citações.
"""

import re
from typing import List, Tuple, Optional


# Padrão para âncoras
_ANCORA_PATTERN = re.compile(r'\[ÂNCORA:\s*"((?:[^"\\]|\\.)*)"\]', re.DOTALL)

# Padrão para citações [N]
_CITATION_PATTERN = re.compile(r'\[(\d+)\]')


def extrair_ancoras_com_citacoes(texto: str) -> List[Tuple[str, Optional[int]]]:
    """
    Extrai âncoras do texto junto com suas citações [N].
    
    Args:
        texto: texto contendo âncoras no formato [ÂNCORA: "texto"] [N]
    
    Returns:
        Lista de tuplas (texto_ancora, numero_citacao)
        Exemplo: [("convergiu após 100 épocas", 2), ("MSE is used", 1)]
    """
    resultados = []
    
    # Procura padrão: [ÂNCORA: "texto"] [N]
    # ou: [ÂNCORA: "texto"] seguido de [N] com até 10 chars de espaço
    pattern = re.compile(
        r'\[ÂNCORA:\s*"((?:[^"\\]|\\.)*)"\]\s*\[(\d+)\]',
        re.DOTALL
    )
    
    for match in pattern.finditer(texto):
        texto_ancora = match.group(1).strip()
        citacao = int(match.group(2))
        
        if len(texto_ancora) >= 10:  # Filtra âncoras muito curtas
            resultados.append((texto_ancora, citacao))
    
    # Também procura âncoras sem citação imediata
    ancoras_sem_citacao = _ANCORA_PATTERN.findall(texto)
    for ancora in ancoras_sem_citacao:
        if len(ancora.strip()) >= 10:
            # Verifica se já foi capturada com citação
            if not any(a[0] == ancora.strip() for a in resultados):
                resultados.append((ancora.strip(), None))
    
    return resultados


def extrair_ancora_principal(bloco: str) -> Optional[str]:
    """
    Extrai a âncora mais relevante (mais longa) de um bloco de texto.
    
    Args:
        bloco: bloco de texto com âncoras
    
    Returns:
        Texto da âncora mais longa, ou None se não houver âncoras
    """
    ancoras = _ANCORA_PATTERN.findall(bloco)
    
    # Filtra âncoras válidas (mínimo 20 chars)
    ancoras_validas = [
        a.strip() for a in ancoras
        if len(a.strip()) >= 20
        and not re.match(r'^[\\\$\{\}\[\]_\^]+', a.strip())
    ]
    
    if not ancoras_validas:
        return None
    
    # Retorna a mais longa (geralmente a mais específica)
    return max(ancoras_validas, key=len)


def extrair_citacao_ancora(texto: str, ancora: str) -> Optional[int]:
    """
    Encontra o número da citação [N] mais próxima de uma âncora específica.
    
    Args:
        texto: texto completo
        ancora: texto da âncora para buscar
    
    Returns:
        Número da citação, ou None se não encontrada
    """
    # Procura pela âncora no texto
    ancora_escaped = re.escape(ancora)
    pattern = re.compile(
        rf'\[ÂNCORA:\s*"{ancora_escaped}"\]\s*\[(\d+)\]',
        re.IGNORECASE
    )
    
    match = pattern.search(texto)
    if match:
        return int(match.group(1))
    
    # Fallback: procura citação próxima à âncora (até 50 chars depois)
    ancora_pos = texto.find(ancora)
    if ancora_pos >= 0:
        trecho_posterior = texto[ancora_pos:ancora_pos + 50]
        cit_match = _CITATION_PATTERN.search(trecho_posterior)
        if cit_match:
            return int(cit_match.group(1))
    
    return None


def limpar_ancoras(texto: str) -> str:
    """
    Remove todas as âncoras do texto, mantendo apenas o conteúdo limpo.
    
    Args:
        texto: texto com âncoras
    
    Returns:
        Texto sem âncoras
    """
    texto_limpo = _ANCORA_PATTERN.sub("", texto)
    texto_limpo = re.sub(r'  +', ' ', texto_limpo)  # Remove espaços duplos
    return texto_limpo.strip()